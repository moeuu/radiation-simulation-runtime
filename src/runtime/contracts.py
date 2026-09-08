"""Shared constants for estimator-neutral simulation artifacts."""

FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY = (
    "full_spectrum_contract_hash_sha256"
)
# Persisted format identity shared with estimators, never a runtime selector.
FULL_SPECTRUM_MODEL_SCHEMA_VERSION = 7

__all__ = [
    "FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY",
    "FULL_SPECTRUM_MODEL_SCHEMA_VERSION",
]
