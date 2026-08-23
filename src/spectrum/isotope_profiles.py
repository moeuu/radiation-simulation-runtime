"""Named isotope selections for decommissioning localization experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping, Sequence

from spectrum.library import require_nuclide


@dataclass(frozen=True)
class IsotopeExperimentProfile:
    """Describe a reusable isotope set and its physical placement policy."""

    name: str
    isotopes: tuple[str, ...]
    material_conditioning: str
    description: str

    def __post_init__(self) -> None:
        """Validate one immutable named experiment profile."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Isotope profile name must be a nonempty string.")
        isotopes = tuple(str(value) for value in self.isotopes)
        if not isotopes or len(set(isotopes)) != len(isotopes):
            raise ValueError("Isotope profile isotopes must be nonempty and unique.")
        for isotope in isotopes:
            require_nuclide(isotope)
        if self.material_conditioning not in {"none", "catalog_physical"}:
            raise ValueError(
                "Isotope profile material_conditioning must be none or "
                "catalog_physical."
            )
        object.__setattr__(self, "isotopes", isotopes)


_PROFILES: Mapping[str, IsotopeExperimentProfile] = MappingProxyType(
    {
        "unconditioned_eu154": IsotopeExperimentProfile(
            name="unconditioned_eu154",
            isotopes=("Cs-137", "Co-60", "Eu-154"),
            material_conditioning="none",
            description="Area-uniform Cs/Co/Eu set without material conditioning.",
        ),
        "fukushima_eu154": IsotopeExperimentProfile(
            name="fukushima_eu154",
            isotopes=("Cs-137", "Co-60", "Eu-154"),
            material_conditioning="catalog_physical",
            description=(
                "Fukushima decommissioning set with Co-60 and Eu-154 "
                "restricted to compatible activated materials."
            ),
        ),
        "fukushima_eu152": IsotopeExperimentProfile(
            name="fukushima_eu152",
            isotopes=("Cs-137", "Co-60", "Eu-152"),
            material_conditioning="catalog_physical",
            description=(
                "Fukushima decommissioning set with Eu-152 restricted to "
                "activated concrete."
            ),
        ),
        "fukushima_nb94": IsotopeExperimentProfile(
            name="fukushima_nb94",
            isotopes=("Cs-137", "Co-60", "Nb-94"),
            material_conditioning="catalog_physical",
            description=(
                "Fukushima decommissioning set with Nb-94 restricted to "
                "activated metal."
            ),
        ),
        "fukushima_cs134": IsotopeExperimentProfile(
            name="fukushima_cs134",
            isotopes=("Cs-137", "Co-60", "Cs-134"),
            material_conditioning="catalog_physical",
            description="Residual radiocesium comparison set.",
        ),
        "fukushima_sb125": IsotopeExperimentProfile(
            name="fukushima_sb125",
            isotopes=("Cs-137", "Co-60", "Sb-125"),
            material_conditioning="catalog_physical",
            description="Fission-product surface-contamination comparison set.",
        ),
        "fukushima_am241": IsotopeExperimentProfile(
            name="fukushima_am241",
            isotopes=("Cs-137", "Co-60", "Am-241"),
            material_conditioning="catalog_physical",
            description="Fuel-debris and actinide-contamination comparison set.",
        ),
    }
)


def available_isotope_profiles() -> tuple[str, ...]:
    """Return all selectable profile names in deterministic order."""
    return tuple(sorted(_PROFILES))


def require_isotope_profile(name: str) -> IsotopeExperimentProfile:
    """Return a named profile or fail with the complete supported set."""
    normalized = str(name).strip().lower()
    try:
        return _PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(available_isotope_profiles())
        raise ValueError(
            f"Unknown isotope_experiment_profile {name!r}; expected: {supported}."
        ) from exc


def resolve_isotope_selection(
    *,
    profile_name: object,
    explicit_isotopes: object,
    fallback_isotopes: Sequence[str],
) -> tuple[tuple[str, ...], IsotopeExperimentProfile | None]:
    """Resolve one profile or explicit list without ambiguous precedence."""
    if profile_name is not None and explicit_isotopes is not None:
        raise ValueError(
            "isotope_experiment_profile and random_source_isotopes are "
            "mutually exclusive."
        )
    if profile_name is not None:
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise TypeError("isotope_experiment_profile must be a JSON string.")
        profile = require_isotope_profile(profile_name)
        return profile.isotopes, profile
    if explicit_isotopes is None:
        names = tuple(str(value) for value in fallback_isotopes)
    elif isinstance(explicit_isotopes, str):
        names = tuple(
            value.strip()
            for value in explicit_isotopes.split(",")
            if value.strip()
        )
    elif isinstance(explicit_isotopes, Sequence):
        if any(not isinstance(value, str) for value in explicit_isotopes):
            raise TypeError(
                "random_source_isotopes entries must be JSON strings."
            )
        names = tuple(value.strip() for value in explicit_isotopes if value.strip())
    else:
        raise TypeError("random_source_isotopes must be a string or sequence.")
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "random_source_isotopes must be nonempty and must not contain "
            "duplicates."
        )
    for isotope in names:
        require_nuclide(isotope)
    return tuple(sorted(names)), None


def resolve_profile_model_runtime_config(
    runtime_config: Mapping[str, object],
    *,
    run_root: str | Path | None = None,
) -> dict[str, object]:
    """Resolve one profile-specific immutable model asset from its registry."""
    resolved: dict[str, object] = dict(runtime_config)
    profile_value = resolved.get("isotope_experiment_profile")
    registry_value = resolved.get("full_spectrum_model_registry_path")
    if profile_value is None and registry_value is None:
        return resolved
    if profile_value is None or registry_value is None:
        raise ValueError(
            "Profile-backed full-spectrum selection requires both "
            "isotope_experiment_profile and full_spectrum_model_registry_path."
        )
    if any(
        key in resolved
        for key in (
            "full_spectrum_generative_model",
            "full_spectrum_generative_model_path",
            "full_spectrum_generative_model_file_sha256",
            "full_spectrum_contract_hash_sha256",
        )
    ):
        raise ValueError(
            "Profile registry selection cannot be combined with an explicit "
            "full-spectrum model."
        )
    profile = require_isotope_profile(str(profile_value))
    registry_path = _resolve_registry_path(registry_value, run_root=run_root)
    raw_bytes = registry_path.read_bytes()
    declared_registry_hash = resolved.get(
        "full_spectrum_model_registry_file_sha256"
    )
    actual_registry_hash = hashlib.sha256(raw_bytes).hexdigest()
    if declared_registry_hash != actual_registry_hash:
        raise ValueError(
            "Full-spectrum model registry SHA-256 does not match its file."
        )
    payload = json.loads(raw_bytes)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("model") != "isotope_profile_full_spectrum_registry"
        or not isinstance(payload.get("profiles"), Mapping)
    ):
        raise ValueError("Full-spectrum model registry payload is invalid.")
    entry = payload["profiles"].get(profile.name)
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"Profile {profile.name!r} has no full-spectrum model asset."
        )
    expected_keys = {
        "isotopes",
        "model_path",
        "model_file_sha256",
        "model_contract_hash_sha256",
        "calibration_status",
    }
    if set(entry) != expected_keys:
        raise ValueError(
            f"Profile registry entry {profile.name!r} has invalid fields."
        )
    if tuple(entry["isotopes"]) != profile.isotopes:
        raise ValueError(
            f"Profile registry isotopes disagree for {profile.name!r}."
        )
    _resolve_registry_asset_path(
        entry["model_path"],
        registry_path=registry_path,
    )
    resolved["full_spectrum_generative_model_path"] = str(entry["model_path"])
    resolved["full_spectrum_generative_model_file_sha256"] = entry[
        "model_file_sha256"
    ]
    resolved["full_spectrum_contract_hash_sha256"] = entry[
        "model_contract_hash_sha256"
    ]
    resolved["full_spectrum_profile_calibration_status"] = entry[
        "calibration_status"
    ]
    return resolved


def _resolve_registry_path(
    value: object,
    *,
    run_root: str | Path | None,
) -> Path:
    """Resolve an authenticated registry without relying on process CWD."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError("full_spectrum_model_registry_path must be a string.")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    roots = []
    if run_root is not None:
        roots.append(Path(run_root).expanduser().resolve())
    roots.append(Path(__file__).resolve().parents[2])
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Full-spectrum registry not found: {value}.")


def _resolve_registry_asset_path(
    value: object,
    *,
    registry_path: Path,
) -> Path:
    """Resolve one registry asset relative to the repository or registry."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Registry model_path must be a nonempty string.")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    for root in (repository_root, registry_path.parent):
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Registered full-spectrum model not found: {value}.")
