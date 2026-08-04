"""Define immutable material and transport-physics table contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


OBSTACLE_MATERIAL_CONTRACT_ID = "known_composite_materials_xcom_v1"
TRANSPORT_PHYSICS_TABLE_CONTRACT_ID = (
    "klein_nishina_xcom_single_scatter_quadrature_v1"
)


def _canonical_sha256(payload: object) -> str:
    """Return the canonical JSON SHA-256 for one physics payload."""
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def obstacle_material_contract_payload() -> dict[str, object]:
    """Return all material values used by analytic obstacle transport."""
    asset = (
        Path(__file__).resolve().parents[1]
        / "sim"
        / "isaacsim_app"
        / "materials.py"
    )
    source_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "contract_id": OBSTACLE_MATERIAL_CONTRACT_ID,
        "mass_attenuation_source": "NIST_XCOM_tabulation_interpolation",
        "material_table_asset": "src/sim/isaacsim_app/materials.py",
        "material_table_source_sha256": source_sha256,
    }


OBSTACLE_MATERIAL_CONTRACT_SHA256 = _canonical_sha256(
    obstacle_material_contract_payload()
)


def transport_physics_table_contract_payload() -> dict[str, object]:
    """Return universal constants and quadrature semantics for transport."""
    return {
        "schema_version": 1,
        "contract_id": TRANSPORT_PHYSICS_TABLE_CONTRACT_ID,
        "electron_rest_energy_keV": 510.99895,
        "classical_electron_radius_cm": 2.8179403262e-13,
        "avogadro_constant_mol_inv": 6.02214076e23,
        "single_scatter_cross_section": "Klein_Nishina_unpolarized",
        "single_scatter_geometry": (
            "detector_cone_conditioned_actual_material_path_quadrature"
        ),
        "scatter_energy_operator": (
            "Klein_Nishina_Gauss_Legendre_column_stochastic"
        ),
        "higher_order_semantics": (
            "positive_scatter_component_nuisance_mean_one"
        ),
        "obstacle_material_contract_sha256": (
            OBSTACLE_MATERIAL_CONTRACT_SHA256
        ),
    }


TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256 = _canonical_sha256(
    transport_physics_table_contract_payload()
)


def require_physics_contracts(payload: Mapping[str, object]) -> None:
    """Reject a payload that does not bind all universal physics contracts."""
    expected = {
        "obstacle_material_contract_id": OBSTACLE_MATERIAL_CONTRACT_ID,
        "obstacle_material_contract_sha256": (
            OBSTACLE_MATERIAL_CONTRACT_SHA256
        ),
        "transport_physics_table_contract_id": (
            TRANSPORT_PHYSICS_TABLE_CONTRACT_ID
        ),
        "transport_physics_table_contract_sha256": (
            TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
        ),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Physics contract field {key!r} is incompatible.")


__all__ = [
    "OBSTACLE_MATERIAL_CONTRACT_ID",
    "OBSTACLE_MATERIAL_CONTRACT_SHA256",
    "TRANSPORT_PHYSICS_TABLE_CONTRACT_ID",
    "TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256",
    "obstacle_material_contract_payload",
    "require_physics_contracts",
    "transport_physics_table_contract_payload",
]
