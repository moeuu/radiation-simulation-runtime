"""Strict JSON framing for machine-readable events on mixed CLI output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from runtime.provenance import strict_json_loads


@dataclass(frozen=True, slots=True)
class CLIJSONEventFraming:
    """Encode and decode one prefixed JSON-object event stream."""

    prefix: str

    def __post_init__(self) -> None:
        """Require an unambiguous single-line prefix."""
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("CLI event prefix must be a nonempty string.")
        if "\n" in self.prefix or "\r" in self.prefix:
            raise ValueError("CLI event prefix must fit on one line.")

    def encode(self, payload: Mapping[str, object]) -> str:
        """Return one deterministic newline-terminated event frame."""
        if not isinstance(payload, Mapping):
            raise TypeError("CLI event payload must be an object.")
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            sort_keys=True,
        )
        return f"{self.prefix}{encoded}\n"

    def parse(self, line: str) -> dict[str, object]:
        """Parse one matching frame with duplicate-key and NaN rejection."""
        if not isinstance(line, str):
            raise TypeError("CLI event frame must be text.")
        normalized = line.removesuffix("\n").removesuffix("\r")
        if not normalized.startswith(self.prefix):
            raise ValueError("CLI event frame does not use the expected prefix.")
        payload = strict_json_loads(normalized.removeprefix(self.prefix))
        if not isinstance(payload, dict):
            raise TypeError("CLI event payload must be an object.")
        return payload

    def try_parse(self, line: str) -> dict[str, object] | None:
        """Return a parsed event, or null for an ordinary diagnostic line."""
        if not isinstance(line, str):
            raise TypeError("CLI output line must be text.")
        normalized = line.removesuffix("\n").removesuffix("\r")
        if not normalized.startswith(self.prefix):
            return None
        return self.parse(normalized)


__all__ = ["CLIJSONEventFraming"]
