#!/usr/bin/env python3
"""Build profile-specific physics-only full-spectrum model assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from measurement.shielding import (
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_CM,
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_FE_SHIELD_THICKNESS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_THICKNESS_CM,
)
from spectrum.additive_scatter import (
    PhysicsOnlyNoncollidedTransportResponse,
)
from spectrum.transport_spectral import (
    GeometryConditionedSpectralModel,
    PhysicalComponentDiscrepancy,
)


def _canonical_bytes(payload: object) -> bytes:
    """Return stable pretty-printed JSON bytes for one asset."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def build_assets(repository_root: Path) -> None:
    """Replace registry profile targets with non-empirical model assets."""
    registry_path = (
        repository_root
        / "configs/geant4/models/isotope_profile_model_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profiles = registry["profiles"]
    response = PhysicsOnlyNoncollidedTransportResponse(
        detector_radius_m=DEFAULT_DETECTOR_CRYSTAL_RADIUS_CM / 100.0,
        fe_scatter_distance_m=(
            DEFAULT_FE_SHIELD_INNER_RADIUS_CM
            + 0.5 * DEFAULT_FE_SHIELD_THICKNESS_CM
        )
        / 100.0,
        pb_scatter_distance_m=(
            DEFAULT_PB_SHIELD_INNER_RADIUS_CM
            + 0.5 * DEFAULT_PB_SHIELD_THICKNESS_CM
        )
        / 100.0,
    )
    for profile_name, entry in sorted(profiles.items()):
        isotopes = tuple(str(value) for value in entry["isotopes"])
        model = GeometryConditionedSpectralModel.standard_native(
            isotopes,
            dead_time_tau_s=5.813e-9,
            background_rate_cps=12.0,
            physical_component_discrepancy=(
                PhysicalComponentDiscrepancy.physics_only_budget()
            ),
            additive_scatter_response=response,
        )
        relative_path = (
            "configs/geant4/models/profiles/"
            f"{profile_name}_physics_only_v4.json"
        )
        model_path = repository_root / relative_path
        model_bytes = _canonical_bytes(model.manifest_payload())
        model_path.write_bytes(model_bytes)
        entry.update(
            {
                "model_path": relative_path,
                "model_file_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "model_contract_hash_sha256": model.contract_hash_sha256,
                "calibration_status": (
                    "physics_only_no_scene_fit_runtime_unvalidated"
                ),
            }
        )
    registry_bytes = _canonical_bytes(registry)
    registry_path.write_bytes(registry_bytes)
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    standard_config_path = (
        repository_root
        / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    standard_config = json.loads(
        standard_config_path.read_text(encoding="utf-8")
    )
    standard_config["full_spectrum_model_registry_file_sha256"] = registry_hash
    standard_config_path.write_bytes(_canonical_bytes(standard_config))


def main() -> None:
    """Build assets relative to this repository."""
    build_assets(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
