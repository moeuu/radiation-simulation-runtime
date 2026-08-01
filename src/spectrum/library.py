"""Nuclear decay metadata and detector-count transport line bases.

``decay_lines`` are evaluated marginal photon probabilities per parent decay.
``lines`` are the detector-count-rate transport basis. The two are kept
separate because an authenticated legacy PF model may fix a coarser line basis.
Prompt cascades are sampled by Geant4 RadioactiveDecay and are never
reconstructed by treating marginal photon probabilities as exclusive branches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class NuclideLine:
    """Represent one gamma energy and a positive line weight."""

    energy_keV: float
    intensity: float

    def __post_init__(self) -> None:
        """Validate a finite positive-energy marginal emission probability."""
        energy = float(self.energy_keV)
        intensity = float(self.intensity)
        if not math.isfinite(energy) or energy <= 0.0:
            raise ValueError("NuclideLine energy_keV must be finite and positive.")
        if not math.isfinite(intensity) or intensity <= 0.0 or intensity > 1.0:
            raise ValueError(
                "NuclideLine intensity must be a photon probability in (0, 1]."
            )
        object.__setattr__(self, "energy_keV", energy)
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True)
class Nuclide:
    """Hold evaluated decay, transport-line, and source-origin metadata."""

    name: str
    lines: Sequence[NuclideLine]
    representative_energy_keV: float
    decay_lines: tuple[NuclideLine, ...] = ()
    atomic_number: int = 0
    mass_number: int = 0
    geant4_excitation_keV: float = 0.0
    half_life_s: float = 0.0
    source_origin: str = "surface_contamination"
    eligible_materials: tuple[str, ...] = ("*",)
    decay_data_reference: str = ""
    prompt_cascade_model: str = "geant4_radioactive_decay"

    def __post_init__(self) -> None:
        """Validate immutable evaluated nuclide metadata."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Nuclide name must be a nonempty string.")
        lines = tuple(self.lines)
        if not lines or any(not isinstance(line, NuclideLine) for line in lines):
            raise ValueError("Nuclide lines must contain NuclideLine entries.")
        energies = tuple(float(line.energy_keV) for line in lines)
        if len(set(energies)) != len(energies):
            raise ValueError("Nuclide gamma energies must be unique.")
        decay_lines = tuple(self.decay_lines) or lines
        if any(not isinstance(line, NuclideLine) for line in decay_lines):
            raise ValueError(
                "Nuclide decay_lines must contain NuclideLine entries."
            )
        decay_energies = tuple(float(line.energy_keV) for line in decay_lines)
        if len(set(decay_energies)) != len(decay_energies):
            raise ValueError("Nuclide decay gamma energies must be unique.")
        representative = float(self.representative_energy_keV)
        if not math.isfinite(representative) or representative <= 0.0:
            raise ValueError(
                "Nuclide representative_energy_keV must be finite and positive."
            )
        if isinstance(self.atomic_number, bool) or int(self.atomic_number) < 0:
            raise ValueError("Nuclide atomic_number must be a nonnegative integer.")
        if isinstance(self.mass_number, bool) or int(self.mass_number) < 0:
            raise ValueError("Nuclide mass_number must be a nonnegative integer.")
        if (int(self.atomic_number) == 0) != (int(self.mass_number) == 0):
            raise ValueError(
                "Nuclide atomic_number and mass_number must either both be "
                "zero or both identify a physical nuclide."
            )
        excitation = float(self.geant4_excitation_keV)
        half_life = float(self.half_life_s)
        if not math.isfinite(excitation) or excitation < 0.0:
            raise ValueError("Nuclide Geant4 excitation must be finite and nonnegative.")
        if not math.isfinite(half_life) or half_life < 0.0:
            raise ValueError("Nuclide half_life_s must be finite and nonnegative.")
        if int(self.atomic_number) > 0 and half_life <= 0.0:
            raise ValueError(
                "A physical Nuclide catalog entry requires a positive half_life_s."
            )
        materials = tuple(str(value).strip().lower() for value in self.eligible_materials)
        if not materials or any(not value for value in materials):
            raise ValueError("Nuclide eligible_materials must be nonempty strings.")
        if "*" in materials and len(materials) != 1:
            raise ValueError("Wildcard material eligibility must be used alone.")
        if self.prompt_cascade_model != "geant4_radioactive_decay":
            raise ValueError(
                "Prompt cascades must use evaluated Geant4 RadioactiveDecay data."
            )
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "decay_lines", decay_lines)
        object.__setattr__(self, "representative_energy_keV", representative)
        object.__setattr__(self, "atomic_number", int(self.atomic_number))
        object.__setattr__(self, "mass_number", int(self.mass_number))
        object.__setattr__(self, "geant4_excitation_keV", excitation)
        object.__setattr__(self, "half_life_s", half_life)
        object.__setattr__(self, "eligible_materials", materials)

    @property
    def mean_gamma_multiplicity(self) -> float:
        """Return the represented mean number of photons per parent decay."""
        return float(sum(line.intensity for line in self.decay_lines))


_DAY_S = 86_400.0
_YEAR_S = 365.25 * _DAY_S


def _line(energy_keV: float, photons_per_decay: float) -> NuclideLine:
    """Construct one concise evaluated gamma-line entry."""
    return NuclideLine(energy_keV=energy_keV, intensity=photons_per_decay)


_NUCLIDES: Mapping[str, Nuclide] = MappingProxyType(
    {
        "Cs-137": Nuclide(
            name="Cs-137",
            lines=[_line(662.0, 0.85)],
            decay_lines=(_line(661.657, 0.8510),),
            representative_energy_keV=662.0,
            atomic_number=55,
            mass_number=137,
            half_life_s=30.018 * _YEAR_S,
            source_origin="accident_surface_contamination",
            eligible_materials=("*",),
            decay_data_reference="LNHB/DDEP 2023 recommended data, Cs-137",
        ),
        "Co-60": Nuclide(
            name="Co-60",
            lines=[
                _line(1173.0, 0.5),
                _line(1332.0, 0.5),
            ],
            decay_lines=(
                _line(1173.228, 0.9985),
                _line(1332.492, 0.999826),
            ),
            representative_energy_keV=1250.0,
            atomic_number=27,
            mass_number=60,
            half_life_s=1925.28 * _DAY_S,
            source_origin="activated_metal_or_concrete",
            eligible_materials=("steel", "iron", "concrete", "aluminum"),
            decay_data_reference="DDEP/Monographie BIPM-5, Co-60",
        ),
        "Eu-154": Nuclide(
            name="Eu-154",
            lines=[
                _line(723.3, 0.25),
                _line(873.2, 0.14),
                _line(996.3, 0.14),
                _line(1274.5, 0.45),
                _line(1494.0, 0.01),
                _line(1596.5, 0.02),
            ],
            decay_lines=(
                _line(123.071, 0.4100),
                _line(247.930, 0.0695),
                _line(401.258, 0.00171),
                _line(591.762, 0.0500),
                _line(692.425, 0.01810),
                _line(723.305, 0.2028),
                _line(873.190, 0.1227),
                _line(996.262, 0.1050),
                _line(1004.725, 0.1817),
                _line(1274.436, 0.3490),
                _line(1494.071, 0.0179),
                _line(1596.481, 0.0178),
            ),
            representative_energy_keV=1274.5,
            atomic_number=63,
            mass_number=154,
            half_life_s=8.601 * _YEAR_S,
            source_origin="activated_concrete",
            eligible_materials=("concrete",),
            decay_data_reference=(
                "IAEA-NDS-112 gamma standards / Geant4 "
                "RadioactiveDecay6.1.2 ENSDF, Eu-154"
            ),
        ),
        "Eu-152": Nuclide(
            name="Eu-152",
            lines=[
                _line(121.782, 0.2841),
                _line(244.697, 0.0755),
                _line(344.279, 0.2658),
                _line(411.117, 0.02237),
                _line(443.965, 0.03125),
                _line(778.905, 0.1296),
                _line(867.380, 0.04241),
                _line(964.079, 0.1462),
                _line(1085.837, 0.1013),
                _line(1089.737, 0.01731),
                _line(1112.076, 0.1340),
                _line(1212.948, 0.01415),
                _line(1299.142, 0.01632),
                _line(1408.013, 0.2085),
            ],
            representative_energy_keV=344.279,
            atomic_number=63,
            mass_number=152,
            # Geant4 RDM stores the 13.5 y state at 45.5998 keV; its zero-keV
            # state is the 9.3 h isomer and must not be selected accidentally.
            geant4_excitation_keV=45.5998,
            half_life_s=13.517 * _YEAR_S,
            source_origin="activated_concrete",
            eligible_materials=("concrete",),
            decay_data_reference="IAEA/DDEP Eu-152 recommended decay data",
        ),
        "Nb-94": Nuclide(
            name="Nb-94",
            lines=[
                _line(702.622, 0.99814),
                _line(871.091, 0.99892),
            ],
            representative_energy_keV=871.091,
            atomic_number=41,
            mass_number=94,
            half_life_s=20_300.0 * _YEAR_S,
            source_origin="activated_metal",
            eligible_materials=("steel", "iron", "stainless_steel", "niobium"),
            decay_data_reference="IAEA INDC(NDS)-0657 / IRDFF-II, Nb-94",
        ),
        "Cs-134": Nuclide(
            name="Cs-134",
            lines=[
                _line(563.246, 0.08342),
                _line(569.331, 0.15368),
                _line(604.721, 0.9763),
                _line(795.864, 0.8547),
                _line(801.953, 0.08694),
                _line(1038.560, 0.00993),
                _line(1167.968, 0.01791),
                _line(1365.185, 0.03014),
            ],
            representative_energy_keV=604.721,
            atomic_number=55,
            mass_number=134,
            half_life_s=2.0644 * _YEAR_S,
            source_origin="accident_surface_contamination",
            eligible_materials=("*",),
            decay_data_reference="IAEA/DDEP Cs-134 recommended decay data",
        ),
        "Sb-125": Nuclide(
            name="Sb-125",
            lines=[
                _line(176.314, 0.0682),
                _line(380.452, 0.0154),
                _line(427.875, 0.2960),
                _line(463.365, 0.1050),
                _line(600.600, 0.1780),
                _line(606.715, 0.0502),
                _line(635.950, 0.1130),
                _line(671.450, 0.0181),
            ],
            representative_energy_keV=427.875,
            atomic_number=51,
            mass_number=125,
            half_life_s=2.75856 * _YEAR_S,
            source_origin="fission_product_surface_contamination",
            eligible_materials=("*",),
            decay_data_reference="DDEP / Geant4 RadioactiveDecay6.1.2, Sb-125",
        ),
        "Am-241": Nuclide(
            name="Am-241",
            lines=[
                _line(26.345, 0.0240),
                _line(33.196, 0.00126),
                _line(43.423, 0.00073),
                _line(59.541, 0.3592),
            ],
            representative_energy_keV=59.541,
            atomic_number=95,
            mass_number=241,
            half_life_s=432.6 * _YEAR_S,
            source_origin="fuel_debris_or_actinide_contamination",
            eligible_materials=("fuel_debris", "corium", "steel", "concrete"),
            decay_data_reference="DDEP Am-241 recommended decay data",
        ),
    }
)


# Positive lines used by the native full-spectrum contract.
KEY_LINES_KEV: Dict[str, List[float]] = {
    isotope: [float(line.energy_keV) for line in nuclide.lines]
    for isotope, nuclide in _NUCLIDES.items()
}


def get_detection_lines_keV(isotope: str) -> List[float]:
    """Return configured positive transport lines in keV."""
    return list(KEY_LINES_KEV.get(isotope, []))


def default_library() -> Dict[str, Nuclide]:
    """Return an independent mapping of supported evaluated nuclides."""
    return dict(_NUCLIDES)


def require_nuclide(isotope: str) -> Nuclide:
    """Return one supported nuclide or fail before simulation starts."""
    try:
        return _NUCLIDES[str(isotope)]
    except KeyError as exc:
        supported = ", ".join(sorted(_NUCLIDES))
        raise ValueError(
            f"Unsupported isotope {isotope!r}; supported isotopes: {supported}."
        ) from exc


def nuclide_catalog_sha256() -> str:
    """Return a deterministic hash of all evaluated catalog semantics."""
    payload = {
        name: {
            "atomic_number": nuclide.atomic_number,
            "mass_number": nuclide.mass_number,
            "geant4_excitation_keV": nuclide.geant4_excitation_keV,
            "half_life_s": nuclide.half_life_s,
            "source_origin": nuclide.source_origin,
            "eligible_materials": list(nuclide.eligible_materials),
            "prompt_cascade_model": nuclide.prompt_cascade_model,
            "decay_data_reference": nuclide.decay_data_reference,
            "lines": [
                {
                    "energy_keV": line.energy_keV,
                    "relative_weight": line.intensity,
                }
                for line in nuclide.lines
            ],
            "decay_lines": [
                {
                    "energy_keV": line.energy_keV,
                    "photons_per_decay": line.intensity,
                }
                for line in nuclide.decay_lines
            ],
        }
        for name, nuclide in sorted(_NUCLIDES.items())
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
