"""Build immutable full-spectrum assets for every isotope experiment profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spectrum.isotope_profiles import available_isotope_profiles, require_isotope_profile
from spectrum.transport_spectral import GeometryConditionedSpectralModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = (
    REPOSITORY_ROOT
    / "configs/geant4/models/geometry_conditioned_full_spectrum_exact_v1.json"
)
DEFAULT_RAL_MODEL = (
    REPOSITORY_ROOT
    / "configs/geant4/models/geometry_conditioned_full_spectrum_ral_eu154_training_v4.json"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "configs/geant4/models/profiles"
DEFAULT_REGISTRY = (
    REPOSITORY_ROOT
    / "configs/geant4/models/isotope_profile_model_registry_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    """Return the profile-model builder command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--ral-model", type=Path, default=DEFAULT_RAL_MODEL)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser


def _canonical_bytes(payload: object) -> bytes:
    """Serialize one JSON payload deterministically for hashing and storage."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_model(path: Path) -> GeometryConditionedSpectralModel:
    """Load and authenticate one source full-spectrum model."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GeometryConditionedSpectralModel.from_manifest_payload(payload)


def _repository_relative(path: Path) -> str:
    """Return one stable repository-relative asset path."""
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def build_profile_models(
    *,
    base_model_path: Path,
    ral_model_path: Path,
    output_directory: Path,
    registry_path: Path,
) -> dict[str, object]:
    """Build physical profile assets and return their registry payload."""
    base_model = _load_model(base_model_path.resolve())
    ral_model = _load_model(ral_model_path.resolve())
    if base_model.additive_scatter_response is None:
        raise ValueError("Base model lacks the physical additive scatter response.")
    output_directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, object] = {}
    generated_by_isotopes: dict[tuple[str, ...], tuple[Path, object]] = {}
    for profile_name in available_isotope_profiles():
        profile = require_isotope_profile(profile_name)
        if profile.isotopes == ("Cs-137", "Co-60", "Eu-154"):
            model_path = ral_model_path.resolve()
            model = ral_model
            calibration_status = (
                "independent_holdout_validated"
                if model.production_ready
                else "training_calibrated_not_independently_validated"
            )
        else:
            cached = generated_by_isotopes.get(profile.isotopes)
            if cached is None:
                model = GeometryConditionedSpectralModel.standard_native(
                    profile.isotopes,
                    dead_time_tau_s=base_model.dead_time_tau_s,
                    background_rate_cps=base_model.background_rate_cps,
                    additive_scatter_response=base_model.additive_scatter_response,
                )
                model_path = output_directory / f"{profile.name}_physical_v1.json"
                model_path.write_bytes(_canonical_bytes(model.manifest_payload()))
                generated_by_isotopes[profile.isotopes] = (model_path, model)
            else:
                model_path, model = cached
            calibration_status = "physical_mean_runtime_unvalidated"
        raw_bytes = model_path.read_bytes()
        entries[profile.name] = {
            "isotopes": list(profile.isotopes),
            "model_path": _repository_relative(model_path),
            "model_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "model_contract_hash_sha256": model.contract_hash_sha256,
            "calibration_status": calibration_status,
        }
    registry = {
        "schema_version": 1,
        "model": "isotope_profile_full_spectrum_registry",
        "profiles": entries,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_bytes(_canonical_bytes(registry))
    return registry


def main() -> None:
    """Build all profile assets and print their immutable digests."""
    arguments = _parser().parse_args()
    registry = build_profile_models(
        base_model_path=arguments.base_model,
        ral_model_path=arguments.ral_model,
        output_directory=arguments.output_directory,
        registry_path=arguments.registry,
    )
    print(f"registry={arguments.registry.resolve()}")
    print(f"registry_sha256={hashlib.sha256(arguments.registry.read_bytes()).hexdigest()}")
    print(f"profiles={len(registry['profiles'])}")


if __name__ == "__main__":
    main()
