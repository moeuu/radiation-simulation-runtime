"""Immutable native Geant4 physics-list and material-resolution contract."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math


GEANT4_PHYSICS_CONTRACT_ID = (
    "geant4_11_3_2_ftfp_bert_em_option4_cut_0p7mm_v1"
)
GEANT4_VERSION_NUMBER = 1132
GEANT4_VERSION_TAG = "geant4-11-03-patch-02"
GEANT4_REFERENCE_PHYSICS_LIST = "FTFP_BERT"
GEANT4_EM_PHYSICS_CONSTRUCTOR = "G4EmStandardPhysics_option4"
GEANT4_PRODUCTION_CUT_RANGE_MM = 0.7
GEANT4_GAMMA_PROCESS_NAMES = "GammaGeneralProc,Transportation"
GEANT4_GAMMA_EM_SUBPROCESS_NAMES = "Rayl,compt,conv,phot"
GEANT4_MATERIAL_RESOLUTION_CONTRACT_ID = (
    "exported_density_mass_composition_except_g4_air_v1"
)


def geant4_physics_contract_payload() -> dict[str, object]:
    """Return the canonical native physics provenance payload."""
    return {
        "contract_id": GEANT4_PHYSICS_CONTRACT_ID,
        "geant4_version_number": GEANT4_VERSION_NUMBER,
        "geant4_version_tag": GEANT4_VERSION_TAG,
        "reference_physics_list": GEANT4_REFERENCE_PHYSICS_LIST,
        "electromagnetic_physics_constructor": (
            GEANT4_EM_PHYSICS_CONSTRUCTOR
        ),
        "production_cut_range_mm": GEANT4_PRODUCTION_CUT_RANGE_MM,
        "gamma_process_names": GEANT4_GAMMA_PROCESS_NAMES,
        "gamma_em_subprocess_names": (
            GEANT4_GAMMA_EM_SUBPROCESS_NAMES
        ),
        "material_resolution_contract_id": (
            GEANT4_MATERIAL_RESOLUTION_CONTRACT_ID
        ),
    }


GEANT4_PHYSICS_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        geant4_physics_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def validate_geant4_physics_metadata(
    metadata: Mapping[str, object],
) -> None:
    """Fail closed unless native metadata matches the full physics contract."""
    expected = {
        **geant4_physics_contract_payload(),
        "contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
    }
    field_map = {
        "contract_id": "geant4_physics_contract_id",
        "contract_sha256": "geant4_physics_contract_sha256",
        "material_resolution_contract_id": (
            "material_resolution_contract_id"
        ),
    }
    for contract_key, expected_value in expected.items():
        metadata_key = field_map.get(contract_key, contract_key)
        actual = metadata.get(metadata_key)
        if isinstance(expected_value, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            ):
                raise RuntimeError(
                    f"Native Geant4 physics metadata {metadata_key} is invalid."
                )
        elif type(actual) is not type(expected_value) or actual != expected_value:
            raise RuntimeError(
                f"Native Geant4 physics metadata {metadata_key} is invalid."
            )


__all__ = [
    "GEANT4_EM_PHYSICS_CONSTRUCTOR",
    "GEANT4_GAMMA_EM_SUBPROCESS_NAMES",
    "GEANT4_GAMMA_PROCESS_NAMES",
    "GEANT4_MATERIAL_RESOLUTION_CONTRACT_ID",
    "GEANT4_PHYSICS_CONTRACT_ID",
    "GEANT4_PHYSICS_CONTRACT_SHA256",
    "GEANT4_PRODUCTION_CUT_RANGE_MM",
    "GEANT4_REFERENCE_PHYSICS_LIST",
    "GEANT4_VERSION_NUMBER",
    "GEANT4_VERSION_TAG",
    "geant4_physics_contract_payload",
    "validate_geant4_physics_metadata",
]
