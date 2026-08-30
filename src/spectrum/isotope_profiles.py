"""Named isotope selections for decommissioning localization experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

from runtime.provenance import strict_json_loads
from spectrum.library import require_nuclide


_REGISTRY_FIELDS = frozenset({"schema_version", "model", "profiles"})
_REGISTRY_ENTRY_FIELDS = frozenset(
    {
        "isotopes",
        "model_path",
        "model_file_sha256",
        "model_contract_hash_sha256",
    }
)


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
        if (
            type(self.isotopes) is not tuple
            or not self.isotopes
            or any(type(value) is not str or not value for value in self.isotopes)
            or len(set(self.isotopes)) != len(self.isotopes)
        ):
            raise ValueError("Isotope profile isotopes must be nonempty and unique.")
        for isotope in self.isotopes:
            require_nuclide(isotope)
        if self.material_conditioning not in {"none", "catalog_physical"}:
            raise ValueError(
                "Isotope profile material_conditioning must be none or "
                "catalog_physical."
            )


_PROFILES: Mapping[str, IsotopeExperimentProfile] = MappingProxyType(
    {
        "unconditioned_cs_co": IsotopeExperimentProfile(
            name="unconditioned_cs_co",
            isotopes=("Cs-137", "Co-60"),
            material_conditioning="none",
            description="Area-uniform Cs/Co set without material conditioning.",
        ),
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
    if type(name) is not str or not name:
        raise TypeError("Isotope profile name must be a nonempty JSON string.")
    try:
        return _PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(available_isotope_profiles())
        raise ValueError(
            f"Unknown isotope_experiment_profile {name!r}; expected: {supported}."
        ) from exc


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
    if type(profile_value) is not str:
        raise TypeError("isotope_experiment_profile must be a JSON string.")
    profile = require_isotope_profile(profile_value)
    registry_path = _resolve_registry_path(registry_value, run_root=run_root)
    raw_bytes = registry_path.read_bytes()
    declared_registry_hash = resolved.get("full_spectrum_model_registry_file_sha256")
    actual_registry_hash = hashlib.sha256(raw_bytes).hexdigest()
    if declared_registry_hash != actual_registry_hash:
        raise ValueError(
            "Full-spectrum model registry SHA-256 does not match its file."
        )
    payload = strict_json_loads(raw_bytes)
    if not isinstance(payload, Mapping) or set(payload) != _REGISTRY_FIELDS:
        raise ValueError("Full-spectrum model registry payload is invalid.")
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("Full-spectrum model registry payload is invalid.")
    if payload["model"] != "isotope_profile_full_spectrum_registry" or not isinstance(
        payload["profiles"], Mapping
    ):
        raise ValueError("Full-spectrum model registry payload is invalid.")
    entry = payload["profiles"].get(profile.name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Profile {profile.name!r} has no full-spectrum model asset.")
    if set(entry) != _REGISTRY_ENTRY_FIELDS:
        raise ValueError(f"Profile registry entry {profile.name!r} has invalid fields.")
    entry_isotopes = entry["isotopes"]
    if (
        type(entry_isotopes) is not list
        or any(type(value) is not str for value in entry_isotopes)
        or tuple(entry_isotopes) != profile.isotopes
    ):
        raise ValueError(f"Profile registry isotopes disagree for {profile.name!r}.")
    for key in (
        "model_path",
        "model_file_sha256",
        "model_contract_hash_sha256",
    ):
        if type(entry[key]) is not str or not entry[key]:
            raise TypeError(
                f"Profile registry entry {profile.name!r} field {key} "
                "must be a nonempty JSON string."
            )
    _resolve_registry_asset_path(
        entry["model_path"],
        registry_path=registry_path,
    )
    resolved["full_spectrum_generative_model_path"] = str(entry["model_path"])
    resolved["full_spectrum_generative_model_file_sha256"] = entry["model_file_sha256"]
    resolved["full_spectrum_contract_hash_sha256"] = entry["model_contract_hash_sha256"]
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
