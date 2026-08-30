"""Owner-only socket contract for evaluation CUI source truth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import socket
from threading import Event, Thread
import time
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from runtime.provenance import strict_canonical_json_bytes, strict_json_loads


_TRUTH_FIELDS = frozenset(
    {
        "schema_version",
        "semantics",
        "true_sources",
        "true_strengths",
    }
)
_RESPONSE_FIELDS = frozenset({"type", "schema_version", "truth"})
_REQUEST = {
    "type": "cui_truth_overlay",
    "schema_version": 1,
}
_SEMANTICS = "evaluation_cui_overlay_only_not_estimator_input"
_MAX_MESSAGE_BYTES = 1024 * 1024


def _finite_positions(value: object, *, isotope: str) -> NDArray[np.float64]:
    """Return one immutable finite N-by-3 truth-position matrix."""
    try:
        positions = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CUI truth positions for {isotope} must be numeric."
        ) from exc
    if positions.size == 0:
        positions = np.zeros((0, 3), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"CUI truth positions for {isotope} must have shape (source, 3)."
        )
    if np.any(~np.isfinite(positions)):
        raise ValueError(f"CUI truth positions for {isotope} must be finite.")
    result = np.array(positions, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_strengths(
    value: object,
    *,
    isotope: str,
    source_count: int,
) -> NDArray[np.float64]:
    """Return one immutable positive strength vector aligned with truth."""
    try:
        strengths = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CUI truth strengths for {isotope} must be numeric."
        ) from exc
    if strengths.shape != (source_count,):
        raise ValueError(
            f"CUI truth strengths for {isotope} must align with its sources."
        )
    if np.any(~np.isfinite(strengths)) or np.any(strengths <= 0.0):
        raise ValueError(
            f"CUI truth strengths for {isotope} must be finite and positive."
        )
    result = np.array(strengths, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CUITruthOverlay:
    """Store validated evaluation truth outside estimator-visible state."""

    true_sources: Mapping[str, NDArray[np.float64]]
    true_strengths: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        """Validate isotope keys, positions, strengths, and alignment."""
        source_keys = set(self.true_sources)
        strength_keys = set(self.true_strengths)
        if source_keys != strength_keys:
            raise ValueError("CUI truth source and strength isotope keys must match.")
        if any(not isinstance(value, str) or not value for value in source_keys):
            raise ValueError("CUI truth isotope keys must be nonempty strings.")
        sources: dict[str, NDArray[np.float64]] = {}
        strengths: dict[str, NDArray[np.float64]] = {}
        for isotope in sorted(source_keys):
            positions = _finite_positions(self.true_sources[isotope], isotope=isotope)
            values = _positive_strengths(
                self.true_strengths[isotope],
                isotope=isotope,
                source_count=int(positions.shape[0]),
            )
            sources[isotope] = positions
            strengths[isotope] = values
        object.__setattr__(self, "true_sources", MappingProxyType(sources))
        object.__setattr__(self, "true_strengths", MappingProxyType(strengths))

    @classmethod
    def from_truth_payload(cls, payload: object) -> "CUITruthOverlay":
        """Parse the exact runtime truth object without accepting aliases."""
        if not isinstance(payload, Mapping) or set(payload) != _TRUTH_FIELDS:
            raise ValueError("CUI truth overlay fields disagree with schema 1.")
        if (
            payload.get("schema_version") != 1
            or payload.get("semantics") != _SEMANTICS
        ):
            raise ValueError("CUI truth overlay contract is invalid.")
        sources = payload.get("true_sources")
        strengths = payload.get("true_strengths")
        if not isinstance(sources, Mapping) or not isinstance(strengths, Mapping):
            raise TypeError("CUI truth sources and strengths must be objects.")
        return cls(true_sources=sources, true_strengths=strengths)

    def to_truth_payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible private truth object."""
        return {
            "schema_version": 1,
            "semantics": _SEMANTICS,
            "true_sources": {
                isotope: values.tolist()
                for isotope, values in self.true_sources.items()
            },
            "true_strengths": {
                isotope: values.tolist()
                for isotope, values in self.true_strengths.items()
            },
        }


def _receive_json_line(connection: socket.socket) -> object:
    """Receive exactly one bounded newline-terminated strict JSON value."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(min(65536, _MAX_MESSAGE_BYTES + 1 - size))
        if not chunk:
            raise EOFError("CUI truth socket closed before a complete message.")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            if chunk[newline + 1 :]:
                raise ValueError("CUI truth socket received trailing message bytes.")
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > _MAX_MESSAGE_BYTES:
            raise ValueError("CUI truth socket message exceeds its size limit.")
    payload = b"".join(chunks)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("CUI truth socket message exceeds its size limit.")
    return strict_json_loads(payload)


def _send_json_line(connection: socket.socket, payload: Mapping[str, object]) -> None:
    """Send one canonical newline-terminated JSON object."""
    normalized = strict_json_loads(strict_canonical_json_bytes(payload))
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    connection.sendall(encoded + b"\n")


class CUITruthOverlaySocketServer:
    """Serve one owner-only truth response to an evaluation renderer."""

    def __init__(
        self,
        socket_path: str | Path,
        truth_payload: Mapping[str, object],
    ) -> None:
        """Bind the private socket synchronously and start its worker thread."""
        self.endpoint = Path(socket_path).expanduser().resolve()
        if self.endpoint.exists() or self.endpoint.is_symlink():
            raise FileExistsError(
                f"CUI truth overlay socket already exists: {self.endpoint}"
            )
        self.endpoint.parent.mkdir(parents=True, exist_ok=True)
        self.overlay = CUITruthOverlay.from_truth_payload(truth_payload)
        self._stop = Event()
        self._failure: BaseException | None = None
        self._served = False
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._server.bind(self.endpoint.as_posix())
            os.chmod(self.endpoint, 0o600)
            self._server.listen(1)
            self._server.settimeout(0.1)
        except BaseException:
            self._server.close()
            self.endpoint.unlink(missing_ok=True)
            raise
        self._thread = Thread(
            target=self._serve,
            name="cui-truth-overlay",
            daemon=True,
        )
        self._thread.start()

    @property
    def served(self) -> bool:
        """Return whether one renderer received the complete overlay."""
        return bool(self._served)

    def _serve(self) -> None:
        """Accept and answer one exact renderer request."""
        try:
            while not self._stop.is_set():
                try:
                    connection, _ = self._server.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._stop.is_set() and exc.errno in {
                        errno.EBADF,
                        errno.EINVAL,
                    }:
                        return
                    raise
                with connection:
                    connection.settimeout(10.0)
                    request = _receive_json_line(connection)
                    if request != _REQUEST:
                        raise ValueError(
                            "CUI truth overlay request disagrees with schema 1."
                        )
                    _send_json_line(
                        connection,
                        {
                            "type": "cui_truth_overlay",
                            "schema_version": 1,
                            "truth": self.overlay.to_truth_payload(),
                        },
                    )
                    self._served = True
                    return
        except BaseException as exc:  # pragma: no cover - surfaced by close/client.
            self._failure = exc

    def close(self) -> None:
        """Stop the private server, remove its socket, and surface failures."""
        self._stop.set()
        self._server.close()
        self._thread.join(timeout=2.0)
        self.endpoint.unlink(missing_ok=True)
        if self._thread.is_alive():
            raise TimeoutError("CUI truth overlay server did not stop.")
        if self._failure is not None:
            raise RuntimeError("CUI truth overlay server failed.") from self._failure


def load_cui_truth_overlay(
    socket_path: str | Path,
    *,
    connect_timeout_s: float = 30.0,
) -> CUITruthOverlay:
    """Load private truth through the renderer-only owner socket."""
    if (
        isinstance(connect_timeout_s, bool)
        or not isinstance(connect_timeout_s, (int, float))
        or not np.isfinite(float(connect_timeout_s))
        or float(connect_timeout_s) <= 0.0
    ):
        raise ValueError("connect_timeout_s must be finite and positive.")
    endpoint = Path(socket_path).expanduser().resolve()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + float(connect_timeout_s)
    try:
        while True:
            try:
                connection.connect(endpoint.as_posix())
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"CUI truth overlay socket was not ready: {endpoint}"
                    ) from None
                time.sleep(0.05)
        connection.settimeout(float(connect_timeout_s))
        _send_json_line(connection, _REQUEST)
        response = _receive_json_line(connection)
    finally:
        connection.close()
    if not isinstance(response, Mapping) or set(response) != _RESPONSE_FIELDS:
        raise ValueError("CUI truth overlay response fields disagree with schema 1.")
    if response.get("type") != "cui_truth_overlay" or response.get(
        "schema_version"
    ) != 1:
        raise ValueError("CUI truth overlay response contract is invalid.")
    return CUITruthOverlay.from_truth_payload(response.get("truth"))


__all__ = [
    "CUITruthOverlay",
    "CUITruthOverlaySocketServer",
    "load_cui_truth_overlay",
]
