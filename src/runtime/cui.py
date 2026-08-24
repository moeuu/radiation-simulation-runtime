"""Truth-free CUI routing and browser server contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
import ipaddress
import os
from pathlib import Path
import socket
import stat
import subprocess
import threading
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import numpy as np
from numpy.typing import NDArray

from runtime.defaults import (
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)
from runtime.measurement_log import MeasurementLogRecord


_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_LOW_PRIORITY_CONTAINER_NETWORK = ipaddress.ip_network("172.16.0.0/12")
CUI_URL_MESSAGE_PREFIX = "CUI split visualization URL:"


def _normalized_host(value: object, *, name: str) -> str:
    """Return one nonempty host without URL syntax or an embedded port."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a nonempty string.")
    host = value.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host or "://" in host or any(char in host for char in "/?#@"):
        raise ValueError(f"{name} must contain a host without URL syntax.")
    if ":" in host:
        address = host.partition("%")[0]
        try:
            ipaddress.IPv6Address(address)
        except ipaddress.AddressValueError as exc:
            raise ValueError(
                f"{name} must not contain an embedded port."
            ) from exc
    return host


def _strict_port(value: object, *, allow_zero: bool) -> int:
    """Return one exact TCP port without accepting booleans or coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("CUI dashboard port must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum or value > 65535:
        qualifier = "0 through 65535" if allow_zero else "1 through 65535"
        raise ValueError(f"CUI dashboard port must be {qualifier}.")
    return int(value)


@dataclass(frozen=True, slots=True)
class CUIDashboardConfig:
    """Configure one truth-free browser-served CUI dashboard."""

    serve: bool = True
    host: str = DEFAULT_CUI_SPLIT_VIEW_HOST
    port: int = DEFAULT_CUI_SPLIT_VIEW_PORT
    public_host: str | None = None

    def __post_init__(self) -> None:
        """Validate CUI options without accepting truthy or numeric coercion."""
        if not isinstance(self.serve, bool):
            raise TypeError("CUI dashboard serve must be a boolean.")
        object.__setattr__(
            self,
            "host",
            _normalized_host(self.host, name="CUI dashboard host"),
        )
        object.__setattr__(
            self,
            "port",
            _strict_port(self.port, allow_zero=True),
        )
        if self.public_host is not None:
            object.__setattr__(
                self,
                "public_host",
                _normalized_host(
                    self.public_host,
                    name="CUI dashboard public_host",
                ),
            )

    @classmethod
    def from_mapping(cls, settings: Mapping[str, object]) -> "CUIDashboardConfig":
        """Parse the existing cross-repository CUI setting names strictly."""
        if not isinstance(settings, Mapping):
            raise TypeError("CUI dashboard settings must be a mapping.")
        return cls(
            serve=settings.get("cui_split_view_serve", True),
            host=settings.get(
                "cui_split_view_host",
                DEFAULT_CUI_SPLIT_VIEW_HOST,
            ),
            port=settings.get(
                "cui_split_view_port",
                DEFAULT_CUI_SPLIT_VIEW_PORT,
            ),
            public_host=settings.get("cui_split_view_public_host"),
        )


def resolve_cui_public_host(
    bind_host: str,
    public_host: str | None = None,
) -> str:
    """Resolve a browser-reachable host using the shared CUI environment rule."""
    normalized_bind = _normalized_host(bind_host, name="CUI dashboard host")
    if public_host is not None:
        return _normalized_host(
            public_host,
            name="CUI dashboard public_host",
        )
    if normalized_bind not in _WILDCARD_HOSTS:
        return normalized_bind
    environment_host = os.environ.get("CUI_SPLIT_VIEW_PUBLIC_HOST")
    if environment_host:
        return _normalized_host(
            environment_host,
            name="CUI_SPLIT_VIEW_PUBLIC_HOST",
        )
    family = socket.AF_INET6 if normalized_bind == "::" else socket.AF_INET
    fallback_address: str | None = None
    try:
        completed = subprocess.run(
            ["hostname", "-I"],
            check=True,
            capture_output=True,
            text=True,
            timeout=0.2,
        )
        candidates = [
            candidate.strip()
            for candidate in completed.stdout.split()
            if candidate.strip()
        ]
        family_candidates = [
            candidate
            for candidate in candidates
            if _usable_discovered_address(candidate, family=family)
        ]
        if family == socket.AF_INET:
            for candidate in family_candidates:
                if candidate.startswith("100."):
                    return candidate
            for candidate in family_candidates:
                address = ipaddress.IPv4Address(candidate)
                if address not in _LOW_PRIORITY_CONTAINER_NETWORK:
                    return candidate
            if family_candidates:
                fallback_address = family_candidates[0]
        else:
            for candidate in family_candidates:
                address = ipaddress.IPv6Address(candidate.partition("%")[0])
                if not address.is_loopback and not address.is_link_local:
                    return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        addresses = socket.getaddrinfo(
            socket.gethostname(),
            None,
            family,
        )
        for address in addresses:
            candidate = str(address[4][0])
            if not _usable_discovered_address(candidate, family=family):
                continue
            if family == socket.AF_INET and candidate.startswith("100."):
                return candidate
            if family == socket.AF_INET6:
                return candidate
            candidate_address = ipaddress.IPv4Address(candidate)
            if candidate_address not in _LOW_PRIORITY_CONTAINER_NETWORK:
                return candidate
            if fallback_address is None:
                fallback_address = candidate
    except OSError:
        pass
    try:
        probe_target: tuple[str, int] = (
            ("2001:4860:4860::8888", 80)
            if family == socket.AF_INET6
            else ("8.8.8.8", 80)
        )
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(probe_target)
            candidate = str(probe.getsockname()[0])
            if _usable_discovered_address(candidate, family=family):
                return candidate
    except OSError:
        pass
    if fallback_address is not None:
        return fallback_address
    return "::1" if family == socket.AF_INET6 else "127.0.0.1"


def _ip_address_family(host: str) -> socket.AddressFamily | None:
    """Return the socket family for one literal IP candidate, if valid."""
    try:
        address = ipaddress.ip_address(host.partition("%")[0])
    except ValueError:
        return None
    return socket.AF_INET6 if address.version == 6 else socket.AF_INET


def _usable_discovered_address(
    candidate: str,
    *,
    family: socket.AddressFamily,
) -> bool:
    """Return whether an automatically discovered address is browser-usable."""
    if _ip_address_family(candidate) != family:
        return False
    address = ipaddress.ip_address(candidate.partition("%")[0])
    return not (
        address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
    )


def cui_browser_url(
    public_host: str,
    port: int,
    index_path: str | Path = "index.html",
) -> str:
    """Build an IPv4-, hostname-, or IPv6-safe CUI dashboard URL."""
    host = _normalized_host(public_host, name="CUI dashboard public_host")
    normalized_port = _strict_port(port, allow_zero=False)
    path = Path(index_path)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in str(index_path)
        or any(part.startswith(".") for part in path.parts)
    ):
        raise ValueError(
            "CUI dashboard index_path must be a visible relative path."
        )
    display_host = host.replace("%", "%25")
    if ":" in display_host:
        display_host = f"[{display_host}]"
    encoded_path = quote(path.as_posix().lstrip("/"), safe="/")
    return f"http://{display_host}:{normalized_port}/{encoded_path}"


def _readonly_float_matrix(
    value: object,
    *,
    name: str,
    minimum_rows: int = 0,
) -> NDArray[np.float64]:
    """Return a finite immutable float64 matrix with three columns."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (N, 3).") from exc
    if (
        array.ndim != 2
        or array.shape[1:] != (3,)
        or array.shape[0] < minimum_rows
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"{name} must have shape (N, 3) with finite values.")
    immutable = np.array(array, dtype=np.float64, copy=True)
    immutable.setflags(write=False)
    return immutable


def _readonly_float_vector(
    value: object,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return one finite immutable float64 vector."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite vector.") from exc
    if array.ndim != 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector.")
    immutable = np.array(array, dtype=np.float64, copy=True)
    immutable.setflags(write=False)
    return immutable


def _readonly_int_vector(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> NDArray[np.int64]:
    """Return one immutable exact integer vector."""
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds the int64 range.")
    array = np.asarray(raw, dtype=np.int64)
    minimum = 1 if positive else 0
    if np.any(array < minimum):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain {qualifier} integers.")
    immutable = np.array(array, dtype=np.int64, copy=True)
    immutable.setflags(write=False)
    return immutable


def _empty_xyz_matrix() -> NDArray[np.float64]:
    """Return a new empty three-dimensional point matrix."""
    return np.zeros((0, 3), dtype=np.float64)


def _empty_int_vector() -> NDArray[np.int64]:
    """Return a new empty exact integer vector."""
    return np.zeros(0, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CUIRoute:
    """Store truth-free route data with one row per runtime station.

    ``measurement_visit_counts`` counts shield-view records acquired at the
    station, matching the compact ``station(view-count)`` PF label.
    ``measurement_step_ids`` stores the first causal step at each station.
    """

    travel_path_segments_xyz: tuple[NDArray[np.float64], ...] = ()
    measurement_stations_xyz: NDArray[np.float64] = field(
        default_factory=_empty_xyz_matrix
    )
    measurement_station_ids: NDArray[np.int64] = field(
        default_factory=_empty_int_vector
    )
    measurement_step_ids: NDArray[np.int64] = field(
        default_factory=_empty_int_vector
    )
    measurement_visit_counts: NDArray[np.int64] = field(
        default_factory=_empty_int_vector
    )
    current_detector_position_xyz: NDArray[np.float64] | None = None
    latest_spectrum_counts: NDArray[np.int64] | None = None
    energy_bin_edges_keV: NDArray[np.float64] | None = None
    latest_step_id: int | None = None

    def __post_init__(self) -> None:
        """Copy and validate every route array as immutable display data."""
        segments = tuple(
            _readonly_float_matrix(
                segment,
                name="travel_path_segments_xyz entry",
                minimum_rows=2,
            )
            for segment in self.travel_path_segments_xyz
        )
        stations = _readonly_float_matrix(
            self.measurement_stations_xyz,
            name="measurement_stations_xyz",
        )
        station_ids = _readonly_int_vector(
            self.measurement_station_ids,
            name="measurement_station_ids",
        )
        step_ids = _readonly_int_vector(
            self.measurement_step_ids,
            name="measurement_step_ids",
        )
        visits = _readonly_int_vector(
            self.measurement_visit_counts,
            name="measurement_visit_counts",
            positive=True,
        )
        if not (
            stations.shape[0]
            == station_ids.size
            == step_ids.size
            == visits.size
        ):
            raise ValueError("CUI measurement station arrays must align.")
        if station_ids.size and np.any(np.diff(station_ids) <= 0):
            raise ValueError(
                "measurement_station_ids must be strictly increasing."
            )
        if step_ids.size and np.any(np.diff(step_ids) <= 0):
            raise ValueError("measurement_step_ids must be strictly increasing.")
        current = None
        if self.current_detector_position_xyz is not None:
            current = _readonly_float_vector(
                self.current_detector_position_xyz,
                name="current_detector_position_xyz",
            )
            if current.shape != (3,):
                raise ValueError(
                    "current_detector_position_xyz must have shape (3,)."
                )
        spectrum = None
        edges = None
        if (self.latest_spectrum_counts is None) is not (
            self.energy_bin_edges_keV is None
        ):
            raise ValueError(
                "latest_spectrum_counts and energy_bin_edges_keV must be set together."
            )
        if self.latest_spectrum_counts is not None:
            spectrum = _readonly_int_vector(
                self.latest_spectrum_counts,
                name="latest_spectrum_counts",
            )
            edges = _readonly_float_vector(
                self.energy_bin_edges_keV,
                name="energy_bin_edges_keV",
            )
            if edges.size != spectrum.size + 1 or np.any(np.diff(edges) <= 0.0):
                raise ValueError(
                    "energy_bin_edges_keV must strictly increase and contain one "
                    "more value than latest_spectrum_counts."
                )
        latest_step_id = self.latest_step_id
        if latest_step_id is not None:
            if isinstance(latest_step_id, bool) or not isinstance(
                latest_step_id,
                (int, np.integer),
            ):
                raise TypeError("latest_step_id must be an integer or null.")
            latest_step_id = int(latest_step_id)
            if latest_step_id < 0:
                raise ValueError("latest_step_id must be non-negative.")
            if step_ids.size and latest_step_id < int(step_ids[-1]):
                raise ValueError(
                    "latest_step_id must not precede the latest station step."
                )
        object.__setattr__(self, "travel_path_segments_xyz", segments)
        object.__setattr__(self, "measurement_stations_xyz", stations)
        object.__setattr__(self, "measurement_station_ids", station_ids)
        object.__setattr__(self, "measurement_step_ids", step_ids)
        object.__setattr__(self, "measurement_visit_counts", visits)
        object.__setattr__(self, "current_detector_position_xyz", current)
        object.__setattr__(self, "latest_spectrum_counts", spectrum)
        object.__setattr__(self, "energy_bin_edges_keV", edges)
        object.__setattr__(self, "latest_step_id", latest_step_id)

    @property
    def path_segments_xyz(self) -> tuple[NDArray[np.float64], ...]:
        """Return the obstacle-aware travel segments under a short alias."""
        return self.travel_path_segments_xyz

    @property
    def measurement_points_xyz(self) -> NDArray[np.float64]:
        """Return measurement stations under the PF display vocabulary."""
        return self.measurement_stations_xyz

    @property
    def current_pose_xyz(self) -> NDArray[np.float64] | None:
        """Return the current detector position under a short alias."""
        return self.current_detector_position_xyz

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-safe renderer payload without realized source truth."""
        stations = [
            {
                "station_id": int(station_id),
                "step_id": int(step_id),
                "position_xyz": position.tolist(),
                "visit_count": int(visit_count),
            }
            for position, station_id, step_id, visit_count in zip(
                self.measurement_stations_xyz,
                self.measurement_station_ids,
                self.measurement_step_ids,
                self.measurement_visit_counts,
                strict=True,
            )
        ]
        return {
            "schema_version": 1,
            "travel_path_segments_xyz": [
                segment.tolist() for segment in self.travel_path_segments_xyz
            ],
            "measurement_stations": stations,
            "current_detector_position_xyz": (
                None
                if self.current_detector_position_xyz is None
                else self.current_detector_position_xyz.tolist()
            ),
            "latest_spectrum_counts": (
                None
                if self.latest_spectrum_counts is None
                else self.latest_spectrum_counts.tolist()
            ),
            "energy_bin_edges_keV": (
                None
                if self.energy_bin_edges_keV is None
                else self.energy_bin_edges_keV.tolist()
            ),
            "latest_step_id": self.latest_step_id,
        }


def _record_waypoints(
    record: MeasurementLogRecord,
) -> NDArray[np.float64] | None:
    """Return one validated truth-free travel segment from record metadata."""
    metadata = getattr(record, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("MeasurementLog record metadata must be a mapping.")
    raw = metadata.get("travel_waypoints_xyz")
    if raw is None:
        return None
    segment = _readonly_float_matrix(
        raw,
        name="record.metadata.travel_waypoints_xyz",
    )
    if segment.shape[0] < 2:
        return None
    return segment


def cui_route_from_records(
    records: Sequence[MeasurementLogRecord],
) -> CUIRoute:
    """Normalize a step-zero causal log prefix into PF-compatible CUI data."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of MeasurementLog records.")
    segments: list[NDArray[np.float64]] = []
    stations: list[NDArray[np.float64]] = []
    station_ids: list[int] = []
    step_ids: list[int] = []
    visit_counts: list[int] = []
    current: NDArray[np.float64] | None = None
    latest_spectrum: NDArray[np.int64] | None = None
    latest_edges: NDArray[np.float64] | None = None
    latest_step_id: int | None = None
    previous_station_id = -1
    for record_index, record in enumerate(records):
        if not isinstance(record, MeasurementLogRecord):
            raise TypeError("records must contain exact MeasurementLogRecord values.")
        if record.step_id != record_index:
            raise ValueError(
                "MeasurementLog record step_id must equal causal record order."
            )
        if record.station_id < previous_station_id:
            raise ValueError("MeasurementLog station_id must be nondecreasing.")
        previous_station_id = record.station_id
        segment = _record_waypoints(record)
        if segment is not None and not (
            segments
            and segments[-1].shape == segment.shape
            and np.array_equal(segments[-1], segment)
        ):
            segments.append(segment)
        current = _readonly_float_vector(
            record.detector_pose_xyz,
            name="record.detector_pose_xyz",
        )
        if current.shape != (3,):
            raise ValueError("record.detector_pose_xyz must have shape (3,).")
        if station_ids and record.station_id == station_ids[-1]:
            if float(np.linalg.norm(current - stations[-1])) > 1.0e-6:
                raise ValueError(
                    "Every record in a CUI measurement station must share one pose."
                )
            visit_counts[-1] += 1
        else:
            stations.append(current)
            station_ids.append(record.station_id)
            step_ids.append(record.step_id)
            visit_counts.append(1)
        latest_spectrum = _readonly_int_vector(
            record.spectrum_counts,
            name="record.spectrum_counts",
        )
        latest_edges = _readonly_float_vector(
            record.energy_bin_edges_keV,
            name="record.energy_bin_edges_keV",
        )
        latest_step_id = record.step_id
    station_matrix = (
        np.vstack(stations)
        if stations
        else np.zeros((0, 3), dtype=np.float64)
    )
    return CUIRoute(
        travel_path_segments_xyz=tuple(segments),
        measurement_stations_xyz=station_matrix,
        measurement_station_ids=np.asarray(station_ids, dtype=np.int64),
        measurement_step_ids=np.asarray(step_ids, dtype=np.int64),
        measurement_visit_counts=np.asarray(visit_counts, dtype=np.int64),
        current_detector_position_xyz=current,
        latest_spectrum_counts=latest_spectrum,
        energy_bin_edges_keV=latest_edges,
        latest_step_id=latest_step_id,
    )


class _SafeQuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve only ordinary root-contained files without directory discovery."""

    def _request_parts(self) -> tuple[str, ...]:
        """Parse one URL into visible root-relative filesystem components."""
        decoded = unquote(urlsplit(self.path).path, errors="strict")
        if "\x00" in decoded or "\\" in decoded:
            raise ValueError("Invalid CUI request path.")
        parts = tuple(part for part in decoded.split("/") if part)
        if not parts or any(
            part in {".", ".."} or part.startswith(".") for part in parts
        ):
            raise ValueError("Hidden or relative CUI request paths are forbidden.")
        return parts

    def _open_request_file(self) -> tuple[Any, str]:
        """Open a regular artifact through no-follow directory descriptors."""
        parts = self._request_parts()
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        directory_descriptor = os.open(
            Path(self.directory).resolve(),
            directory_flags,
        )
        try:
            for part in parts[:-1]:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
                os.close(directory_descriptor)
                directory_descriptor = child_descriptor
            file_descriptor = os.open(
                parts[-1],
                file_flags,
                dir_fd=directory_descriptor,
            )
            try:
                file_status = os.fstat(file_descriptor)
                if not stat.S_ISREG(file_status.st_mode):
                    raise ValueError("CUI request target is not a regular file.")
                return os.fdopen(file_descriptor, "rb"), parts[-1]
            except BaseException:
                os.close(file_descriptor)
                raise
        finally:
            os.close(directory_descriptor)

    def send_head(self) -> Any:
        """Open one safe regular file and reject every directory request."""
        try:
            handle, filename = self._open_request_file()
        except (OSError, UnicodeError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        try:
            status = os.fstat(handle.fileno())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", self.guess_type(filename))
            self.send_header("Content-Length", str(status.st_size))
            self.send_header("Last-Modified", self.date_time_string(status.st_mtime))
            self.end_headers()
            return handle
        except BaseException:
            handle.close()
            raise

    def log_request(
        self,
        code: int | str = "-",
        size: int | str = "-",
    ) -> None:
        """Suppress unbounded per-request access logging for the CUI poller."""
        del code, size

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress request-error messages emitted by the HTTP base class."""
        del format, args


class _CUIThreadingHTTPServer(ThreadingHTTPServer):
    """Release request threads promptly when a managed CUI server closes."""

    daemon_threads = True


class _IPv6CUIThreadingHTTPServer(_CUIThreadingHTTPServer):
    """Bind a CUI HTTP server using an IPv6 socket."""

    address_family = socket.AF_INET6


def _server_class(host: str) -> type[_CUIThreadingHTTPServer]:
    """Return the HTTP server class matching the configured address family."""
    literal_family = _ip_address_family(host)
    if literal_family == socket.AF_INET6:
        return _IPv6CUIThreadingHTTPServer
    if literal_family == socket.AF_INET:
        return _CUIThreadingHTTPServer
    try:
        families = {
            result[0]
            for result in socket.getaddrinfo(
                host,
                None,
                type=socket.SOCK_STREAM,
            )
        }
    except OSError:
        return _CUIThreadingHTTPServer
    if families == {socket.AF_INET6}:
        return _IPv6CUIThreadingHTTPServer
    return _CUIThreadingHTTPServer


def _build_http_server(
    root: Path,
    host: str,
    port: int,
) -> _CUIThreadingHTTPServer:
    """Bind one safe quiet threaded HTTP server to the requested static root."""
    handler = partial(_SafeQuietHTTPRequestHandler, directory=root.as_posix())
    return _server_class(host)((host, port), handler)


def _contains_symlink(root: Path, relative_path: Path) -> bool:
    """Return whether an existing component below root is a symbolic link."""
    cursor = root
    for part in relative_path.parts:
        cursor /= part
        if cursor.is_symlink():
            return True
    return False


def _resolved_root_and_index(
    root: str | Path,
    index_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Resolve a static root and require its browser index to remain inside it."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"CUI dashboard root does not exist: {root_path}")
    candidate = Path(index_path).expanduser()
    if not candidate.is_absolute():
        candidate = root_path / candidate
    lexical_index = Path(os.path.abspath(candidate))
    try:
        lexical_relative = lexical_index.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("CUI dashboard index_path must remain inside root.") from exc
    if any(part.startswith(".") for part in lexical_relative.parts):
        raise ValueError("CUI dashboard index_path must not contain dotfiles.")
    if _contains_symlink(root_path, lexical_relative):
        raise ValueError("CUI dashboard index_path must not contain symlinks.")
    resolved_index = lexical_index.resolve()
    try:
        relative_index = resolved_index.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("CUI dashboard index_path must remain inside root.") from exc
    if resolved_index.exists() and not resolved_index.is_file():
        raise ValueError(
            f"CUI dashboard index is not a regular file: {resolved_index}"
        )
    return root_path, resolved_index, relative_index


@dataclass(slots=True)
class CUIServerHandle:
    """Own one in-process CUI HTTP server with deterministic shutdown."""

    root: Path
    index_path: Path
    host: str
    port: int | None
    url: str | None
    persistent: bool = False
    _server: _CUIThreadingHTTPServer | None = field(
        default=None,
        repr=False,
    )
    _thread: threading.Thread | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def managed(self) -> bool:
        """Return whether this handle directly owns an in-process server."""
        return self._server is not None

    @property
    def process_id(self) -> int | None:
        """Return no process identifier because every server is in-process."""
        return None

    def close(self) -> None:
        """Stop this handle's in-process server and release its bound port."""
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def terminate(self) -> None:
        """Stop this handle using the same deterministic path as close."""
        self.close()

    def __enter__(self) -> "CUIServerHandle":
        """Return this server handle for managed context usage."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close only a context-managed in-process server."""
        del exc_type, exc_value, traceback
        self.close()


def _start_fixed_port_server(
    root: Path,
    index_path: Path,
    relative_index: Path,
    config: CUIDashboardConfig,
) -> CUIServerHandle:
    """Bind the first available fixed port without probing another listener."""
    display_host = resolve_cui_public_host(config.host, config.public_host)
    for port in range(config.port, min(65535, config.port + 99) + 1):
        try:
            server = _build_http_server(root, config.host, port)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
        return _start_server_thread(
            server,
            root=root,
            index_path=index_path,
            relative_index=relative_index,
            host=config.host,
            display_host=display_host,
        )
    raise RuntimeError(
        "No available CUI dashboard TCP port was found in the configured range."
    )


def _start_server_thread(
    server: _CUIThreadingHTTPServer,
    *,
    root: Path,
    index_path: Path,
    relative_index: Path,
    host: str,
    display_host: str,
) -> CUIServerHandle:
    """Start one already-bound managed server and return its owning handle."""
    actual_port = int(server.server_address[1])
    thread = threading.Thread(
        target=server.serve_forever,
        name="cui-dashboard-http-server",
        daemon=True,
    )
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise
    return CUIServerHandle(
        root=root,
        index_path=index_path,
        host=host,
        port=actual_port,
        url=cui_browser_url(display_host, actual_port, relative_index),
        _server=server,
        _thread=thread,
    )


def start_cui_server(
    root: str | Path,
    *,
    index_path: str | Path = "index.html",
    config: CUIDashboardConfig | None = None,
) -> CUIServerHandle:
    """Start a managed CUI server on an ephemeral or first available fixed port."""
    resolved_config = CUIDashboardConfig() if config is None else config
    if not isinstance(resolved_config, CUIDashboardConfig):
        raise TypeError("config must be a CUIDashboardConfig.")
    root_path, resolved_index, relative_index = _resolved_root_and_index(
        root,
        index_path,
    )
    if not resolved_config.serve:
        return CUIServerHandle(
            root=root_path,
            index_path=resolved_index,
            host=resolved_config.host,
            port=None,
            url=None,
        )
    if resolved_config.port != 0:
        return _start_fixed_port_server(
            root_path,
            resolved_index,
            relative_index,
            resolved_config,
        )
    server = _build_http_server(root_path, resolved_config.host, 0)
    display_host = resolve_cui_public_host(
        resolved_config.host,
        resolved_config.public_host,
    )
    return _start_server_thread(
        server,
        root=root_path,
        index_path=resolved_index,
        host=resolved_config.host,
        relative_index=relative_index,
        display_host=display_host,
    )


__all__ = [
    "CUI_URL_MESSAGE_PREFIX",
    "CUIDashboardConfig",
    "CUIRoute",
    "CUIServerHandle",
    "cui_browser_url",
    "cui_route_from_records",
    "resolve_cui_public_host",
    "start_cui_server",
]
