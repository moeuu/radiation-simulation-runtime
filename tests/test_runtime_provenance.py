"""Tests for the single fail-closed runtime JSON contract."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import numpy as np
import runtime.provenance as provenance

from runtime.provenance import (
    strict_canonical_json_bytes,
    strict_sha256_json,
)


def test_strict_json_serializer_is_deterministic() -> None:
    """The sole public artifact serializer must produce deterministic bytes."""
    payload = {"name": "run", "values": (1, 2.5, None, True)}

    strict = strict_canonical_json_bytes(payload)

    assert strict_sha256_json(payload) == sha256(strict).hexdigest()
    assert not hasattr(provenance, "canonical_json_bytes")
    assert not hasattr(provenance, "sha256_json")
    assert not hasattr(provenance, "json_safe")


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        ({1: "integer", "1": "string"}, TypeError),
        ({"value": object()}, TypeError),
        ({"value": np.int64(3)}, TypeError),
        ({"value": Path("artifact.json")}, TypeError),
        ({"value": float("inf")}, ValueError),
    ),
)
def test_json_serializers_reject_lossy_coercions(
    payload: object,
    error: type[Exception],
) -> None:
    """Artifact identities must fail instead of changing value semantics."""
    with pytest.raises(error):
        strict_canonical_json_bytes(payload)
