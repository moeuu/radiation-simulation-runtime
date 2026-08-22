"""Tests for estimator-neutral strict JSON and config identity helpers."""

from __future__ import annotations

import hashlib

import pytest

from runtime.provenance import (
    ConfigIdentity,
    load_strict_json,
    strict_canonical_json_bytes,
    strict_json_loads,
)


def test_strict_json_loader_rejects_ambiguous_documents(tmp_path) -> None:
    """Duplicate keys, non-finite values, and invalid UTF-8 must fail closed."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": 1, "x": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        load_strict_json(duplicate)
    with pytest.raises(ValueError, match="NaN"):
        strict_json_loads('{"x": NaN}')
    with pytest.raises(ValueError, match="UTF-8"):
        strict_json_loads(b"\xff")


def test_config_identity_distinguishes_source_and_effective_hashes(tmp_path) -> None:
    """Source formatting and resolved defaults must have separate identities."""
    source = tmp_path / "config.json"
    source_bytes = b'{"mode": "live"}\n'
    source.write_bytes(source_bytes)
    effective = {"mode": "live", "timeout_s": 30}

    identity = ConfigIdentity.from_path(source, effective)

    assert identity.source_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert identity.effective_sha256 == hashlib.sha256(
        strict_canonical_json_bytes(effective)
    ).hexdigest()
    assert ConfigIdentity.from_payload(identity.to_payload()) == identity
    assert ConfigIdentity.from_source_bytes(
        b'{ "mode": "live" }\n',
        effective,
    ).source_sha256 != identity.source_sha256
    assert ConfigIdentity.from_source_bytes(
        source_bytes,
        {**effective, "timeout_s": 31},
    ).effective_sha256 != identity.effective_sha256


def test_config_identity_rejects_lossy_effective_values() -> None:
    """Effective configurations must remain strict JSON-native data."""
    with pytest.raises(TypeError, match="unsupported JSON value"):
        ConfigIdentity.from_source_bytes(b"{}", {"path": object()})
