"""Universal dry-air photon attenuation from the NIST XCOM reference table."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from numpy.typing import NDArray


G4_AIR_REFERENCE_DENSITY_G_CM3 = 1.20479e-3
NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID = "nist_xcom_dry_air_total_v1"

# NIST XCOM total mass attenuation coefficients for dry air.  The energy
# interval covers every photon line supported by the experiment profiles.
NIST_XCOM_DRY_AIR_ENERGY_KEV = np.asarray(
    (
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        80.0,
        100.0,
        150.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        800.0,
        1000.0,
        1250.0,
        1500.0,
        2000.0,
    ),
    dtype=np.float64,
)
NIST_XCOM_DRY_AIR_TOTAL_MASS_ATTENUATION_CM2_G = np.asarray(
    (
        0.7779,
        0.3538,
        0.2485,
        0.2080,
        0.1875,
        0.1662,
        0.1541,
        0.1356,
        0.1233,
        0.1067,
        0.09549,
        0.08712,
        0.08055,
        0.07074,
        0.06358,
        0.05687,
        0.05175,
        0.04447,
    ),
    dtype=np.float64,
)


def _contract_payload() -> dict[str, object]:
    """Return the immutable dry-air attenuation contract payload."""
    return {
        "contract_id": NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
        "source": "NIST_XCOM_dry_air_total_attenuation",
        "interpolation": "log_energy_log_coefficient_linear",
        "density_g_cm3": G4_AIR_REFERENCE_DENSITY_G_CM3,
        "energy_keV": NIST_XCOM_DRY_AIR_ENERGY_KEV.tolist(),
        "total_mass_attenuation_cm2_g": (
            NIST_XCOM_DRY_AIR_TOTAL_MASS_ATTENUATION_CM2_G.tolist()
        ),
    }


NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def dry_air_total_mass_attenuation_numpy(
    energy_keV: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Return NIST XCOM dry-air total mass attenuation in cm^2/g."""
    energy = np.asarray(energy_keV, dtype=np.float64)
    if (
        np.any(~np.isfinite(energy))
        or np.any(energy < NIST_XCOM_DRY_AIR_ENERGY_KEV[0])
        or np.any(energy > NIST_XCOM_DRY_AIR_ENERGY_KEV[-1])
    ):
        raise ValueError(
            "Dry-air attenuation energy is outside the authenticated XCOM table."
        )
    return np.exp(
        np.interp(
            np.log(energy),
            np.log(NIST_XCOM_DRY_AIR_ENERGY_KEV),
            np.log(NIST_XCOM_DRY_AIR_TOTAL_MASS_ATTENUATION_CM2_G),
        )
    )


def dry_air_total_linear_attenuation_numpy(
    energy_keV: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Return dry-air total linear attenuation in inverse centimetres."""
    return (
        dry_air_total_mass_attenuation_numpy(energy_keV)
        * G4_AIR_REFERENCE_DENSITY_G_CM3
    )


def dry_air_total_linear_attenuation_torch(energy_keV: object) -> object:
    """Return the batched Torch dry-air attenuation in inverse centimetres."""
    import torch

    energy = torch.as_tensor(energy_keV)
    if energy.dtype != torch.float64:
        raise TypeError("Production dry-air attenuation requires torch.float64.")
    if (
        bool(torch.any(~torch.isfinite(energy)))
        or bool(torch.any(energy < NIST_XCOM_DRY_AIR_ENERGY_KEV[0]))
        or bool(torch.any(energy > NIST_XCOM_DRY_AIR_ENERGY_KEV[-1]))
    ):
        raise ValueError(
            "Dry-air attenuation energy is outside the authenticated XCOM table."
        )
    grid_energy = torch.as_tensor(
        NIST_XCOM_DRY_AIR_ENERGY_KEV,
        device=energy.device,
        dtype=energy.dtype,
    )
    grid_mu = torch.as_tensor(
        NIST_XCOM_DRY_AIR_TOTAL_MASS_ATTENUATION_CM2_G,
        device=energy.device,
        dtype=energy.dtype,
    )
    log_energy = torch.log(energy)
    log_grid = torch.log(grid_energy)
    upper = torch.searchsorted(log_grid, log_energy, right=True)
    upper = torch.clamp(upper, min=1, max=grid_energy.numel() - 1)
    lower = upper - 1
    lower_energy = log_grid[lower]
    upper_energy = log_grid[upper]
    fraction = (log_energy - lower_energy) / (upper_energy - lower_energy)
    log_mu = torch.log(grid_mu)
    mass_mu = torch.exp(log_mu[lower] + fraction * (log_mu[upper] - log_mu[lower]))
    return mass_mu * G4_AIR_REFERENCE_DENSITY_G_CM3
