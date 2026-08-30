"""Shared names of runtime keys that select one spectral model artifact."""

from __future__ import annotations

from typing import Final


FULL_SPECTRUM_MODEL_RUNTIME_KEYS: Final = frozenset(
    {
        "full_spectrum_generative_model",
        "full_spectrum_generative_model_path",
        "full_spectrum_generative_model_file_sha256",
        "full_spectrum_contract_hash_sha256",
        "full_spectrum_model_registry_file_sha256",
        "full_spectrum_model_registry_path",
        "isotope_experiment_profile",
    }
)


__all__ = ["FULL_SPECTRUM_MODEL_RUNTIME_KEYS"]
