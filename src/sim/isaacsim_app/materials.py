"""Material presets and attenuation helpers for the Isaac Sim bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class MaterialPreset:
    """Describe a reusable material preset for physical attenuation lookup."""

    name: str
    density_g_cm3: float | None = None
    mass_att_by_isotope_cm2_g: dict[str, float] = field(default_factory=dict)
    composition_by_mass: dict[str, float] = field(default_factory=dict)


# XCOM total mass attenuation coefficients in cm^2/g for the gamma-energy
# range used by the runtime isotopes. Values are from NIST X-Ray Mass
# Attenuation Coefficients, Table 3:
# https://physics.nist.gov/PhysRefData/XrayMassCoef/tab3.html
# The 20--400 keV rows are required by the committed Eu-152, Sb-125, and
# Am-241 transport-line contracts. Production callers reject extrapolation.
ELEMENTAL_MASS_ATT_CURVES_CM2_G: dict[str, dict[float, float]] = {
    "H": {
        20.0: 0.36950,
        30.0: 0.35700,
        40.0: 0.34580,
        50.0: 0.33550,
        60.0: 0.32600,
        80.0: 0.30910,
        100.0: 0.29440,
        150.0: 0.26510,
        200.0: 0.24290,
        300.0: 0.21120,
        400.0: 0.18930,
        500.0: 0.17290,
        600.0: 0.15990,
        800.0: 0.14050,
        1000.0: 0.12630,
        1250.0: 0.11290,
        1500.0: 0.10270,
        2000.0: 0.08769,
    },
    "C": {
        20.0: 0.44200,
        30.0: 0.25620,
        40.0: 0.20760,
        50.0: 0.18710,
        60.0: 0.17530,
        80.0: 0.16100,
        100.0: 0.15140,
        150.0: 0.13470,
        200.0: 0.12290,
        300.0: 0.10660,
        400.0: 0.09546,
        500.0: 0.08715,
        600.0: 0.08058,
        800.0: 0.07076,
        1000.0: 0.06361,
        1250.0: 0.05690,
        1500.0: 0.05179,
        2000.0: 0.04442,
    },
    "N": {
        20.0: 0.61780,
        30.0: 0.30660,
        40.0: 0.22880,
        50.0: 0.19800,
        60.0: 0.18170,
        80.0: 0.16390,
        100.0: 0.15290,
        150.0: 0.13530,
        200.0: 0.12330,
        300.0: 0.10680,
        400.0: 0.09557,
        500.0: 0.08719,
        600.0: 0.08063,
        800.0: 0.07081,
        1000.0: 0.06364,
        1250.0: 0.05693,
        1500.0: 0.05180,
        2000.0: 0.04450,
    },
    "O": {
        20.0: 0.86510,
        30.0: 0.37790,
        40.0: 0.25850,
        50.0: 0.21320,
        60.0: 0.19070,
        80.0: 0.16780,
        100.0: 0.15510,
        150.0: 0.13610,
        200.0: 0.12370,
        300.0: 0.10700,
        400.0: 0.09566,
        500.0: 0.08729,
        600.0: 0.08070,
        800.0: 0.07087,
        1000.0: 0.06372,
        1250.0: 0.05697,
        1500.0: 0.05185,
        2000.0: 0.04459,
    },
    "Al": {
        20.0: 3.44100,
        30.0: 1.12800,
        40.0: 0.56850,
        50.0: 0.36810,
        60.0: 0.27780,
        80.0: 0.20180,
        100.0: 0.17040,
        150.0: 0.13780,
        200.0: 0.12230,
        300.0: 0.10420,
        400.0: 0.09276,
        500.0: 0.08445,
        600.0: 0.07802,
        800.0: 0.06841,
        1000.0: 0.06146,
        1250.0: 0.05496,
        1500.0: 0.05006,
        2000.0: 0.04324,
    },
    "Si": {
        20.0: 4.46400,
        30.0: 1.43600,
        40.0: 0.70120,
        50.0: 0.43850,
        60.0: 0.32070,
        80.0: 0.22280,
        100.0: 0.18350,
        150.0: 0.14480,
        200.0: 0.12750,
        300.0: 0.10820,
        400.0: 0.09614,
        500.0: 0.08748,
        600.0: 0.08077,
        800.0: 0.07082,
        1000.0: 0.06361,
        1250.0: 0.05688,
        1500.0: 0.05183,
        2000.0: 0.04480,
    },
    "Ca": {
        20.0: 13.0600,
        30.0: 4.08000,
        40.0: 1.83000,
        50.0: 1.01900,
        60.0: 0.65780,
        80.0: 0.36560,
        100.0: 0.25710,
        150.0: 0.16740,
        200.0: 0.13760,
        300.0: 0.11160,
        400.0: 0.09783,
        500.0: 0.08851,
        600.0: 0.08148,
        800.0: 0.07122,
        1000.0: 0.06388,
        1250.0: 0.05709,
        1500.0: 0.05207,
        2000.0: 0.04524,
    },
    "Cr": {
        20.0: 20.3800,
        30.0: 6.43400,
        40.0: 2.85600,
        50.0: 1.55000,
        60.0: 0.96390,
        80.0: 0.49050,
        100.0: 0.31660,
        150.0: 0.17880,
        200.0: 0.13780,
        300.0: 0.10670,
        400.0: 0.09213,
        500.0: 0.08281,
        600.0: 0.07598,
        800.0: 0.06620,
        1000.0: 0.05930,
        1250.0: 0.05295,
        1500.0: 0.04832,
        2000.0: 0.04213,
    },
    "Fe": {
        20.0: 25.6800,
        30.0: 8.17600,
        40.0: 3.62900,
        50.0: 1.95800,
        60.0: 1.20500,
        80.0: 0.59520,
        100.0: 0.37170,
        150.0: 0.19640,
        200.0: 0.14600,
        300.0: 0.10990,
        400.0: 0.09400,
        500.0: 0.08414,
        600.0: 0.07704,
        800.0: 0.06699,
        1000.0: 0.05995,
        1250.0: 0.05350,
        1500.0: 0.04883,
        2000.0: 0.04265,
    },
    "Ni": {
        20.0: 32.2000,
        30.0: 10.3400,
        40.0: 4.60000,
        50.0: 2.47400,
        60.0: 1.51200,
        80.0: 0.73060,
        100.0: 0.44400,
        150.0: 0.22080,
        200.0: 0.15820,
        300.0: 0.11540,
        400.0: 0.09765,
        500.0: 0.08698,
        600.0: 0.07944,
        800.0: 0.06891,
        1000.0: 0.06160,
        1250.0: 0.05494,
        1500.0: 0.05015,
        2000.0: 0.04387,
    },
    "Ar": {
        20.0: 8.62900,
        30.0: 2.69700,
        40.0: 1.22800,
        50.0: 0.70120,
        60.0: 0.46640,
        80.0: 0.27600,
        100.0: 0.20430,
        150.0: 0.14270,
        200.0: 0.12050,
        300.0: 0.09953,
        400.0: 0.08776,
        500.0: 0.07958,
        600.0: 0.07335,
        800.0: 0.06419,
        1000.0: 0.05762,
        1250.0: 0.05150,
        1500.0: 0.04695,
        2000.0: 0.04074,
    },
    "Pb": {
        20.0: 86.3600,
        30.0: 30.3200,
        40.0: 14.3600,
        50.0: 8.04100,
        60.0: 5.02100,
        80.0: 2.41900,
        100.0: 5.54900,
        150.0: 2.01400,
        200.0: 0.99850,
        300.0: 0.40310,
        400.0: 0.23230,
        500.0: 0.16140,
        600.0: 0.12480,
        800.0: 0.08870,
        1000.0: 0.07102,
        1250.0: 0.05876,
        1500.0: 0.05222,
        2000.0: 0.04606,
    },
}

ELEMENTAL_MASS_ATT_CM2_G: dict[str, dict[str, float]] = {
    "H": {"Cs-137": 0.153886, "Co-60": 0.113291, "Eu-154": 0.125769},
    "C": {"Cs-137": 0.077536, "Co-60": 0.057095, "Eu-154": 0.063368},
    "N": {"Cs-137": 0.077586, "Co-60": 0.057122, "Eu-154": 0.063403},
    "O": {"Cs-137": 0.077653, "Co-60": 0.057170, "Eu-154": 0.063460},
    "Al": {"Cs-137": 0.075041, "Co-60": 0.055157, "Eu-154": 0.061248},
    "Si": {"Cs-137": 0.077685, "Co-60": 0.057088, "Eu-154": 0.063397},
    "Ca": {"Cs-137": 0.078299, "Co-60": 0.057312, "Eu-154": 0.063715},
    "Cr": {"Cs-137": 0.072948, "Co-60": 0.053169, "Eu-154": 0.059180},
    "Fe": {"Cs-137": 0.073924, "Co-60": 0.053727, "Eu-154": 0.059853},
    "Ni": {"Cs-137": 0.076176, "Co-60": 0.055180, "Eu-154": 0.061531},
    "Ar": {"Cs-137": 0.070510, "Co-60": 0.051696, "Eu-154": 0.057445},
    "Pb": {"Cs-137": 0.113609, "Co-60": 0.059575, "Eu-154": 0.074094},
}

MATERIAL_PRESETS: dict[str, MaterialPreset] = {
    "air": MaterialPreset(
        name="air",
        density_g_cm3=0.001225,
        composition_by_mass={"N": 0.755, "O": 0.232, "Ar": 0.013},
    ),
    "water": MaterialPreset(
        name="water",
        density_g_cm3=1.0,
        composition_by_mass={"H": 0.1119, "O": 0.8881},
    ),
    "concrete": MaterialPreset(
        name="concrete",
        density_g_cm3=2.3,
        composition_by_mass={"O": 0.525, "Si": 0.325, "Ca": 0.090, "Al": 0.060},
    ),
    "aluminum": MaterialPreset(
        name="aluminum",
        density_g_cm3=2.7,
        composition_by_mass={"Al": 1.0},
    ),
    "iron": MaterialPreset(
        name="iron",
        density_g_cm3=7.87,
        composition_by_mass={"Fe": 1.0},
    ),
    "steel": MaterialPreset(
        name="steel",
        density_g_cm3=7.85,
        composition_by_mass={"Fe": 0.98, "C": 0.02},
    ),
    "stainless_steel": MaterialPreset(
        name="stainless_steel",
        density_g_cm3=8.0,
        composition_by_mass={"Fe": 0.70, "Cr": 0.19, "Ni": 0.10, "C": 0.01},
    ),
    "lead": MaterialPreset(
        name="lead",
        density_g_cm3=11.34,
        composition_by_mass={"Pb": 1.0},
    ),
}

MATERIAL_ALIASES: dict[str, str] = {
    "al": "aluminum",
    "alu": "aluminum",
    "aluminium": "aluminum",
    "fe": "iron",
    "iron": "iron",
    "lead": "lead",
    "pb": "lead",
    "ss": "stainless_steel",
    "ss304": "stainless_steel",
    "stainless": "stainless_steel",
}

ELEMENT_ALIASES: dict[str, str] = {
    "al": "Al",
    "aluminum": "Al",
    "aluminium": "Al",
    "ar": "Ar",
    "argon": "Ar",
    "c": "C",
    "carbon": "C",
    "ca": "Ca",
    "calcium": "Ca",
    "cr": "Cr",
    "chromium": "Cr",
    "fe": "Fe",
    "iron": "Fe",
    "h": "H",
    "hydrogen": "H",
    "n": "N",
    "nitrogen": "N",
    "ni": "Ni",
    "nickel": "Ni",
    "o": "O",
    "oxygen": "O",
    "pb": "Pb",
    "lead": "Pb",
    "si": "Si",
    "silicon": "Si",
}


def normalize_material_name(name: str) -> str:
    """Normalize a material identifier into a preset lookup key."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return MATERIAL_ALIASES.get(normalized, normalized)


def normalize_element_name(name: str) -> str | None:
    """Normalize an element token into its symbol."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return ELEMENT_ALIASES.get(normalized)


def parse_composition_string(raw_value: str) -> dict[str, float]:
    """Parse a simple mass-fraction composition string."""
    composition: dict[str, float] = {}
    text = str(raw_value).strip()
    if not text:
        return composition
    normalized = text.replace(";", ",")
    for part in normalized.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(f"Invalid composition entry: {item!r}")
        element = normalize_element_name(key)
        if element is None:
            continue
        composition[element] = float(value)
    return composition


def normalize_composition_by_mass(raw_value: dict[str, float] | str | None) -> dict[str, float]:
    """Normalize a composition dictionary or composition string."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, str):
        return parse_composition_string(raw_value)
    composition: dict[str, float] = {}
    for key, value in raw_value.items():
        element = normalize_element_name(str(key))
        if element is None:
            continue
        composition[element] = float(value)
    return composition


def resolve_material_preset(name: str | None) -> MaterialPreset | None:
    """Resolve a normalized preset entry by material name."""
    if name in (None, ""):
        return None
    return MATERIAL_PRESETS.get(normalize_material_name(str(name)))


def composition_mass_attenuation(
    composition_by_mass: dict[str, float] | str | None,
    isotope: str,
) -> float | None:
    """Return a mixture mass attenuation coefficient from elemental fractions."""
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for element, weight in composition.items():
        table = ELEMENTAL_MASS_ATT_CM2_G.get(element)
        if table is None:
            continue
        mass_att = table.get(isotope)
        if mass_att is None:
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0.0:
            continue
        total_weight += numeric_weight
        weighted_mu += numeric_weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return weighted_mu / total_weight


def interpolate_mass_attenuation_curve(curve_by_energy_keV: dict[float, float], energy_keV: float) -> float | None:
    """Interpolate a mass attenuation coefficient from a discrete energy curve."""
    if not curve_by_energy_keV:
        return None
    energies = np.asarray(sorted(float(energy) for energy in curve_by_energy_keV.keys()), dtype=float)
    values = np.asarray([float(curve_by_energy_keV[float(energy)]) for energy in energies], dtype=float)
    if energies.size == 0:
        return None
    if energies.size == 1:
        return float(values[0])
    clamped_energy_keV = float(np.clip(float(energy_keV), float(energies[0]), float(energies[-1])))
    return float(np.interp(clamped_energy_keV, energies, values))


def composition_mass_attenuation_at_energy(
    composition_by_mass: dict[str, float] | str | None,
    energy_keV: float,
) -> float | None:
    """Return a mixture mass attenuation coefficient at a specific photon energy."""
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for element, weight in composition.items():
        curve = ELEMENTAL_MASS_ATT_CURVES_CM2_G.get(element)
        if curve is None:
            continue
        mass_att = interpolate_mass_attenuation_curve(curve, energy_keV)
        if mass_att is None:
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0.0:
            continue
        total_weight += numeric_weight
        weighted_mu += numeric_weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return weighted_mu / total_weight


def require_composition_mass_attenuation_at_energy(
    composition_by_mass: dict[str, float] | str | None,
    energy_keV: float,
) -> float:
    """Return an in-range XCOM mixture coefficient or fail explicitly."""
    if isinstance(energy_keV, bool) or not isinstance(energy_keV, Real):
        raise TypeError("energy_keV must be a real number.")
    energy = float(energy_keV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_keV must be finite and positive.")
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        raise ValueError("Material composition must be nonempty.")
    for element, weight in composition.items():
        numeric_weight = float(weight)
        if not np.isfinite(numeric_weight) or numeric_weight < 0.0:
            raise ValueError("Material mass fractions must be finite and nonnegative.")
        if numeric_weight == 0.0:
            continue
        curve = ELEMENTAL_MASS_ATT_CURVES_CM2_G.get(element)
        if curve is None:
            raise ValueError(f"No XCOM attenuation curve exists for {element!r}.")
        lower = min(float(value) for value in curve)
        upper = max(float(value) for value in curve)
        if not lower <= energy <= upper:
            raise ValueError(
                f"energy_keV={energy} is outside the XCOM range "
                f"[{lower}, {upper}] for {element!r}."
            )
    result = composition_mass_attenuation_at_energy(composition, energy)
    if result is None or not np.isfinite(result) or result <= 0.0:
        raise ValueError("No positive XCOM attenuation coefficient is available.")
    return float(result)
