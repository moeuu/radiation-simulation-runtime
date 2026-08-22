"""Tests for legacy-compatible and strict runtime provenance JSON."""

from __future__ import annotations

from hashlib import sha256

import pytest

from runtime.provenance import (
    canonical_json_bytes,
    strict_canonical_json_bytes,
    strict_sha256_json,
)


def test_strict_json_matches_legacy_bytes_for_supported_values() -> None:
    """New schemas should retain stable bytes for ordinary JSON-native data."""
    payload = {"name": "run", "values": (1, 2.5, None, True)}

    strict = strict_canonical_json_bytes(payload)

    assert strict == canonical_json_bytes(payload)
    assert strict_sha256_json(payload) == sha256(strict).hexdigest()


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        ({1: "integer", "1": "string"}, TypeError),
        ({"value": object()}, TypeError),
        ({"value": float("inf")}, ValueError),
    ),
)
def test_strict_json_rejects_legacy_lossy_coercions(
    payload: object,
    error: type[Exception],
) -> None:
    """New artifact identities must fail instead of changing value semantics."""
    with pytest.raises(error):
        strict_canonical_json_bytes(payload)
