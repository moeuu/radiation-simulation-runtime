"""Compare evaluated Geant4 decay cascades with detector-cps line bases.

The comparison is deliberately separate from production observations.  It
uses unit-weight Geant4 histories in both arms, compares conditional detector
pulse spectra rather than incompatible source-rate normalizations, and writes
enough provenance to reproduce every case.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from sim.geant4_app.engine import Geant4StepRequest, validate_native_scene_identity
from sim.geant4_app.io_format import (
    read_response_file,
    write_request_file,
    write_scene_file,
)
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_COINCIDENCE_WINDOW_S,
    ExportedDetectorModel,
    ExportedGeant4Scene,
    ExportedGeant4Source,
)
from sim.isaacsim_app.scene_builder import StagePrimPaths
from spectrum.library import Nuclide, nuclide_catalog_sha256, require_nuclide


_NATIVE_BIN_WIDTH_KEV = 2.0
_STANDARD_PF_ENERGY_MAX_KEV = 1700.0
_COMPARISON_SCHEMA_VERSION = 1
_COMPARISON_DOMAIN = b"decay-cascade-comparison-v1\0"


def _canonical_json_sha256(payload: object) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return one finite JSON number after strict validation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _strict_integer(
    value: object,
    *,
    field_name: str,
    minimum: int = 1,
) -> int:
    """Return one strict JSON integer at or above ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a JSON integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return int(value)


@dataclass(frozen=True)
class DecayCascadeComparisonDesign:
    """Describe one immutable RDM-versus-line-basis diagnostic design."""

    isotopes: tuple[str, ...]
    distances_m: tuple[float, ...]
    target_expected_gamma_intersections: int
    maximum_parent_decays_per_case: int
    independent_line_histories_per_case: int
    energy_max_keV: float
    comparison_bin_width_keV: float
    minimum_rdm_detected_pulses: int
    maximum_common_band_tv: float
    maximum_coincidence_excess_fraction: float
    bootstrap_samples: int
    seed: int
    total_geant4_threads: int
    case_workers: int
    timeout_s: float
    physics_profile: str = "balanced"
    minimum_sum_line_probability: float = 0.01

    def __post_init__(self) -> None:
        """Validate physical ranges and deterministic parallel allocation."""
        isotopes = tuple(str(value).strip() for value in self.isotopes)
        if not isotopes or len(set(isotopes)) != len(isotopes):
            raise ValueError("Comparison isotopes must be nonempty and unique.")
        for isotope in isotopes:
            require_nuclide(isotope)
        distances = tuple(float(value) for value in self.distances_m)
        if (
            not distances
            or len(set(distances)) != len(distances)
            or any(not math.isfinite(value) or value <= 0.0 for value in distances)
        ):
            raise ValueError(
                "Comparison distances must be unique finite positive values."
            )
        if tuple(sorted(distances)) != distances:
            raise ValueError("Comparison distances must be strictly increasing.")
        if self.target_expected_gamma_intersections <= 0:
            raise ValueError("Expected gamma-intersection target must be positive.")
        if self.maximum_parent_decays_per_case <= 0:
            raise ValueError("Maximum parent-decay count must be positive.")
        if self.independent_line_histories_per_case <= 0:
            raise ValueError("Independent-line history count must be positive.")
        if (
            not math.isfinite(self.energy_max_keV)
            or self.energy_max_keV < _STANDARD_PF_ENERGY_MAX_KEV
            or self.energy_max_keV > 10_000.0
            or not math.isclose(
                self.energy_max_keV / _NATIVE_BIN_WIDTH_KEV,
                round(self.energy_max_keV / _NATIVE_BIN_WIDTH_KEV),
            )
        ):
            raise ValueError(
                "Diagnostic energy maximum must be 2-keV aligned in "
                "[1700, 10000] keV."
            )
        if (
            not math.isfinite(self.comparison_bin_width_keV)
            or self.comparison_bin_width_keV < _NATIVE_BIN_WIDTH_KEV
            or not math.isclose(
                self.comparison_bin_width_keV / _NATIVE_BIN_WIDTH_KEV,
                round(self.comparison_bin_width_keV / _NATIVE_BIN_WIDTH_KEV),
            )
        ):
            raise ValueError(
                "Comparison bin width must be a positive multiple of 2 keV."
            )
        if self.minimum_rdm_detected_pulses <= 0:
            raise ValueError("Minimum detected-pulse count must be positive.")
        for name, value in (
            ("maximum_common_band_tv", self.maximum_common_band_tv),
            (
                "maximum_coincidence_excess_fraction",
                self.maximum_coincidence_excess_fraction,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one.")
        if self.bootstrap_samples < 128:
            raise ValueError("At least 128 bootstrap samples are required.")
        if self.total_geant4_threads <= 0 or self.case_workers <= 0:
            raise ValueError("Thread and worker counts must be positive.")
        if self.case_workers > self.total_geant4_threads:
            raise ValueError("Case workers must not exceed total Geant4 threads.")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("Per-case timeout must be finite and positive.")
        if not self.physics_profile.strip():
            raise ValueError("Physics profile must be nonempty.")
        if (
            not math.isfinite(self.minimum_sum_line_probability)
            or self.minimum_sum_line_probability <= 0.0
            or self.minimum_sum_line_probability > 1.0
        ):
            raise ValueError(
                "Minimum sum-line probability must lie in (0, 1]."
            )
        object.__setattr__(self, "isotopes", isotopes)
        object.__setattr__(self, "distances_m", distances)

    @property
    def threads_per_case(self) -> int:
        """Return the non-oversubscribing Geant4 thread count per worker."""
        return max(1, self.total_geant4_threads // self.case_workers)

    def to_dict(self) -> dict[str, object]:
        """Return the complete immutable design payload."""
        return {
            "schema_version": _COMPARISON_SCHEMA_VERSION,
            "isotopes": list(self.isotopes),
            "distances_m": list(self.distances_m),
            "target_expected_gamma_intersections": (
                self.target_expected_gamma_intersections
            ),
            "maximum_parent_decays_per_case": (
                self.maximum_parent_decays_per_case
            ),
            "independent_line_histories_per_case": (
                self.independent_line_histories_per_case
            ),
            "energy_max_keV": self.energy_max_keV,
            "comparison_bin_width_keV": self.comparison_bin_width_keV,
            "minimum_rdm_detected_pulses": self.minimum_rdm_detected_pulses,
            "maximum_common_band_tv": self.maximum_common_band_tv,
            "maximum_coincidence_excess_fraction": (
                self.maximum_coincidence_excess_fraction
            ),
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "total_geant4_threads": self.total_geant4_threads,
            "case_workers": self.case_workers,
            "threads_per_case": self.threads_per_case,
            "timeout_s": self.timeout_s,
            "physics_profile": self.physics_profile,
            "minimum_sum_line_probability": self.minimum_sum_line_probability,
        }

    @property
    def design_sha256(self) -> str:
        """Return the immutable comparison-design digest."""
        return _canonical_json_sha256(self.to_dict())


def load_decay_cascade_comparison_design(
    path: str | Path,
) -> DecayCascadeComparisonDesign:
    """Load and strictly validate one comparison-design JSON file."""
    design_path = Path(path).expanduser().resolve()
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Decay comparison design must be a JSON object.")
    schema = _strict_integer(
        payload.get("schema_version"),
        field_name="schema_version",
    )
    if schema != _COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported comparison schema {schema}; expected "
            f"{_COMPARISON_SCHEMA_VERSION}."
        )
    isotopes = payload.get("isotopes")
    distances = payload.get("distances_m")
    if not isinstance(isotopes, list) or any(
        not isinstance(value, str) for value in isotopes
    ):
        raise TypeError("isotopes must be a JSON string list.")
    if not isinstance(distances, list):
        raise TypeError("distances_m must be a JSON number list.")
    return DecayCascadeComparisonDesign(
        isotopes=tuple(isotopes),
        distances_m=tuple(
            _strict_number(
                value,
                field_name="distances_m[]",
                strictly_positive=True,
            )
            for value in distances
        ),
        target_expected_gamma_intersections=_strict_integer(
            payload.get("target_expected_gamma_intersections"),
            field_name="target_expected_gamma_intersections",
        ),
        maximum_parent_decays_per_case=_strict_integer(
            payload.get("maximum_parent_decays_per_case"),
            field_name="maximum_parent_decays_per_case",
        ),
        independent_line_histories_per_case=_strict_integer(
            payload.get("independent_line_histories_per_case"),
            field_name="independent_line_histories_per_case",
        ),
        energy_max_keV=_strict_number(
            payload.get("energy_max_keV"),
            field_name="energy_max_keV",
            minimum=_STANDARD_PF_ENERGY_MAX_KEV,
        ),
        comparison_bin_width_keV=_strict_number(
            payload.get("comparison_bin_width_keV"),
            field_name="comparison_bin_width_keV",
            strictly_positive=True,
        ),
        minimum_rdm_detected_pulses=_strict_integer(
            payload.get("minimum_rdm_detected_pulses"),
            field_name="minimum_rdm_detected_pulses",
        ),
        maximum_common_band_tv=_strict_number(
            payload.get("maximum_common_band_tv"),
            field_name="maximum_common_band_tv",
            strictly_positive=True,
        ),
        maximum_coincidence_excess_fraction=_strict_number(
            payload.get("maximum_coincidence_excess_fraction"),
            field_name="maximum_coincidence_excess_fraction",
            strictly_positive=True,
        ),
        bootstrap_samples=_strict_integer(
            payload.get("bootstrap_samples"),
            field_name="bootstrap_samples",
            minimum=128,
        ),
        seed=_strict_integer(
            payload.get("seed"),
            field_name="seed",
            minimum=0,
        ),
        total_geant4_threads=_strict_integer(
            payload.get("total_geant4_threads"),
            field_name="total_geant4_threads",
        ),
        case_workers=_strict_integer(
            payload.get("case_workers"),
            field_name="case_workers",
        ),
        timeout_s=_strict_number(
            payload.get("timeout_s"),
            field_name="timeout_s",
            strictly_positive=True,
        ),
        physics_profile=str(payload.get("physics_profile", "balanced")),
        minimum_sum_line_probability=_strict_number(
            payload.get("minimum_sum_line_probability", 0.01),
            field_name="minimum_sum_line_probability",
            strictly_positive=True,
        ),
    )


def detector_reference_acceptance(crystal_radius_m: float) -> float:
    """Return the exact native 1-m circular-target solid-angle fraction."""
    radius = _strict_number(
        crystal_radius_m,
        field_name="crystal_radius_m",
        strictly_positive=True,
    )
    return float(
        np.clip(sphere_solid_angle_fraction(1.0, radius), 1.0e-12, 1.0)
    )


def sphere_solid_angle_fraction(distance_m: float, radius_m: float) -> float:
    """Return the exact solid-angle fraction of the spherical detector target."""
    distance = _strict_number(
        distance_m,
        field_name="distance_m",
        strictly_positive=True,
    )
    radius = _strict_number(
        radius_m,
        field_name="radius_m",
        strictly_positive=True,
    )
    if distance <= radius:
        return 0.5
    ratio = min(1.0, radius / distance)
    return float(0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - ratio * ratio))))


def planned_parent_decays(
    design: DecayCascadeComparisonDesign,
    *,
    isotope: str,
    distance_m: float,
    detector_radius_m: float,
) -> int:
    """Return a predeclared RDM quota based only on physical geometry."""
    nuclide = require_nuclide(isotope)
    expected_intersections_per_decay = (
        nuclide.mean_gamma_multiplicity
        * sphere_solid_angle_fraction(distance_m, detector_radius_m)
    )
    if expected_intersections_per_decay <= 0.0:
        raise ValueError("Expected gamma intersections must be positive.")
    required = math.ceil(
        design.target_expected_gamma_intersections
        / expected_intersections_per_decay
    )
    return min(required, design.maximum_parent_decays_per_case)


def _detector_cps_geometry_scale(distance_m: float, radius_m: float) -> float:
    """Reproduce the native detector-cps geometry normalization exactly."""
    reference = sphere_solid_angle_fraction(max(1.0, radius_m), radius_m)
    current = sphere_solid_angle_fraction(max(distance_m, radius_m), radius_m)
    return current / max(reference, 1.0e-12)


def _case_seed(
    design: DecayCascadeComparisonDesign,
    *,
    isotope: str,
    distance_m: float,
    emission_model: str,
) -> int:
    """Return a deterministic independent seed for one comparison case."""
    payload = (
        _COMPARISON_DOMAIN
        + str(design.seed).encode("ascii")
        + b"\0"
        + isotope.encode("utf-8")
        + b"\0"
        + format(distance_m, ".17g").encode("ascii")
        + b"\0"
        + emission_model.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _comparison_scene(
    *,
    isotope: str,
    distance_m: float,
    source_strength: float,
    detector_model: ExportedDetectorModel,
    design_sha256: str,
    emission_model: str,
) -> ExportedGeant4Scene:
    """Build an empty-air scene that isolates decay and detector semantics."""
    epsilon = SURFACE_EMISSION_EPSILON_M
    source = ExportedGeant4Source(
        isotope=isotope,
        position_xyz=(epsilon, 1.0, 1.0),
        anchor_position_xyz=(0.0, 1.0, 1.0),
        intensity_cps_1m=(
            None
            if emission_model == "geant4_radioactive_decay"
            else float(source_strength)
        ),
        activity_bq=(
            float(source_strength)
            if emission_model == "geant4_radioactive_decay"
            else None
        ),
        surface_chart_id=0,
        surface_uv=(0.5, 0.5),
        surface_normal_xyz=(1.0, 0.0, 0.0),
        surface_emission_policy_sha256=surface_emission_policy_sha256(),
    )
    stable = {
        "design_sha256": design_sha256,
        "isotope": isotope,
        "distance_m": distance_m,
        "emission_model": emission_model,
        "source_strength": source_strength,
        "source_strength_field": (
            "activity_bq"
            if emission_model == "geant4_radioactive_decay"
            else "intensity_cps_1m"
        ),
        "detector": detector_model.to_dict(),
    }
    return ExportedGeant4Scene(
        scene_hash=_canonical_json_sha256(stable),
        usd_path=None,
        room_size_xyz=(max(2.0, distance_m + 1.0), 2.0, 2.0),
        static_volumes=(),
        sources=(source,),
        detector_model=detector_model,
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )


def _comparison_request(*, distance_m: float, seed: int) -> Geant4StepRequest:
    """Build one zero-background, zero-dead-time diagnostic request."""
    detector = (
        float(distance_m + SURFACE_EMISSION_EPSILON_M),
        1.0,
        1.0,
    )
    quaternion = (1.0, 0.0, 0.0, 0.0)
    return Geant4StepRequest(
        step_id=0,
        dwell_time_s=1.0,
        seed=int(seed),
        detector_pose_xyz=detector,
        detector_quat_wxyz=quaternion,
        fe_shield_pose_xyz=detector,
        fe_shield_quat_wxyz=quaternion,
        pb_shield_pose_xyz=detector,
        pb_shield_quat_wxyz=quaternion,
    )


@dataclass(frozen=True)
class _ComparisonCase:
    """Hold all process-serializable inputs for one native comparison case."""

    design: DecayCascadeComparisonDesign
    executable: str
    output_directory: str
    detector_payload: dict[str, object]
    isotope: str
    distance_m: float
    emission_model: str


def _detector_from_payload(payload: Mapping[str, object]) -> ExportedDetectorModel:
    """Reconstruct one validated detector model inside a worker process."""
    return ExportedDetectorModel(
        crystal_radius_m=float(payload["crystal_radius_m"]),
        crystal_length_m=float(payload["crystal_length_m"]),
        housing_thickness_m=float(payload["housing_thickness_m"]),
        coincidence_window_s=float(payload["coincidence_window_s"]),
        crystal_shape=str(payload["crystal_shape"]),
        crystal_material=str(payload["crystal_material"]),
        housing_material=str(payload["housing_material"]),
    )


def _run_comparison_case(case: _ComparisonCase) -> dict[str, object]:
    """Run one isolated native Geant4 case in a worker process."""
    design = case.design
    detector = _detector_from_payload(case.detector_payload)
    output = Path(case.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if case.emission_model == "geant4_radioactive_decay":
        planned_histories = planned_parent_decays(
            design,
            isotope=case.isotope,
            distance_m=case.distance_m,
            detector_radius_m=detector.crystal_radius_m,
        )
        source_strength = float(planned_histories)
        source_rate_model = "parent_decay_activity_bq"
        source_bias_mode = "analog"
    elif case.emission_model == "independent_gamma_lines":
        planned_histories = design.independent_line_histories_per_case
        scale = _detector_cps_geometry_scale(
            case.distance_m,
            detector.crystal_radius_m,
        )
        source_strength = planned_histories / scale
        source_rate_model = "detector_cps_1m"
        source_bias_mode = "detector_cone"
    else:
        raise ValueError(f"Unsupported comparison emission model {case.emission_model}.")
    scene = _comparison_scene(
        isotope=case.isotope,
        distance_m=case.distance_m,
        source_strength=source_strength,
        detector_model=detector,
        design_sha256=design.design_sha256,
        emission_model=case.emission_model,
    )
    seed = _case_seed(
        design,
        isotope=case.isotope,
        distance_m=case.distance_m,
        emission_model=case.emission_model,
    )
    request = _comparison_request(distance_m=case.distance_m, seed=seed)
    scene_path = output / "scene.txt"
    request_path = output / "request.txt"
    response_path = output / "response.txt"
    write_scene_file(scene, scene_path)
    write_request_file(request, request_path)
    command = [
        case.executable,
        "--scene",
        scene_path.as_posix(),
        "--request",
        request_path.as_posix(),
        "--response",
        response_path.as_posix(),
        "--physics-profile",
        design.physics_profile,
        "--threads",
        str(design.threads_per_case),
        "--source-rate-model",
        source_rate_model,
        "--primary-emission-model",
        case.emission_model,
        "--source-bias-mode",
        source_bias_mode,
        "--detector-scoring-mode",
        "full_transport",
        "--secondary-transport-mode",
        "full_transport",
        "--primary-sampling-fraction",
        "1",
        "--background-cps",
        "0",
        "--dead-time-tau-s",
        "0",
        "--decay-comparison-diagnostic",
        "--decay-comparison-energy-max-kev",
        format(design.energy_max_keV, ".17g"),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=design.timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Native decay comparison failed: "
            f"returncode={completed.returncode} stderr={completed.stderr[-4000:]}"
        )
    spectrum, metadata = read_response_file(response_path)
    expected_bins = int(design.energy_max_keV / _NATIVE_BIN_WIDTH_KEV) + 1
    if spectrum.shape != (expected_bins,):
        raise RuntimeError(
            f"Diagnostic spectrum shape {spectrum.shape} != {(expected_bins,)}."
        )
    if metadata.get("decay_comparison_diagnostic") is not True:
        raise RuntimeError("Native response lacks decay-comparison provenance.")
    if metadata.get("source_rate_model") != source_rate_model:
        raise RuntimeError("Native source-rate provenance is inconsistent.")
    if metadata.get("primary_emission_model") != case.emission_model:
        raise RuntimeError("Native primary-emission provenance is inconsistent.")
    if not np.isclose(
        float(metadata.get("primary_history_weight", math.nan)),
        1.0,
    ):
        raise RuntimeError("Decay comparison must use unit-weight histories.")
    validate_native_scene_identity(dict(metadata), scene)
    spectrum_path = output / "spectrum.npy"
    np.save(spectrum_path, np.asarray(spectrum, dtype=np.float64))
    return {
        "isotope": case.isotope,
        "distance_m": case.distance_m,
        "emission_model": case.emission_model,
        "seed": seed,
        "planned_histories": planned_histories,
        "realized_primaries": int(metadata["num_primaries"]),
        "detected_pulses": int(round(float(np.sum(spectrum)))),
        "source_rate_model": source_rate_model,
        "scene_hash": scene.scene_hash,
        "spectrum_path": spectrum_path.as_posix(),
        "response_path": response_path.as_posix(),
        "command": command,
    }


def _rebin_common_band(
    spectrum: NDArray[np.float64],
    *,
    comparison_bin_width_keV: float,
) -> NDArray[np.float64]:
    """Rebin the standard 0--1700 keV PF band with vectorized accumulation."""
    energies = np.arange(spectrum.size, dtype=np.float64) * _NATIVE_BIN_WIDTH_KEV
    mask = energies <= _STANDARD_PF_ENERGY_MAX_KEV
    bin_count = int(
        math.ceil(_STANDARD_PF_ENERGY_MAX_KEV / comparison_bin_width_keV)
    )
    indices = np.minimum(
        (energies[mask] / comparison_bin_width_keV).astype(np.int64),
        bin_count - 1,
    )
    return np.bincount(
        indices,
        weights=np.asarray(spectrum[mask], dtype=np.float64),
        minlength=bin_count,
    )


def _normalized(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a normalized nonnegative vector or fail on an empty spectrum."""
    vector = np.asarray(values, dtype=np.float64)
    total = float(np.sum(vector))
    if np.any(~np.isfinite(vector)) or np.any(vector < 0.0) or total <= 0.0:
        raise ValueError("Comparison spectrum must contain positive finite mass.")
    return vector / total


def _bootstrap_tv_interval(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return point and percentile-bootstrap total-variation distances."""
    first_counts = np.rint(first).astype(np.int64)
    second_counts = np.rint(second).astype(np.int64)
    first_total = int(np.sum(first_counts))
    second_total = int(np.sum(second_counts))
    if first_total <= 0 or second_total <= 0:
        raise ValueError("TV bootstrap requires two nonempty integer spectra.")
    first_probability = _normalized(first_counts.astype(np.float64))
    second_probability = _normalized(second_counts.astype(np.float64))
    point = float(0.5 * np.sum(np.abs(first_probability - second_probability)))
    rng = np.random.default_rng(seed)
    first_draws = rng.multinomial(first_total, first_probability, size=samples)
    second_draws = rng.multinomial(second_total, second_probability, size=samples)
    distances = 0.5 * np.sum(
        np.abs(
            first_draws / float(first_total)
            - second_draws / float(second_total)
        ),
        axis=1,
    )
    lower, upper = np.quantile(distances, (0.025, 0.975))
    return point, float(lower), float(upper)


def _beta_difference_interval(
    first_successes: int,
    first_total: int,
    second_successes: int,
    second_total: int,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return a Jeffreys-posterior interval for a binomial fraction excess."""
    if first_total <= 0 or second_total <= 0:
        raise ValueError("Fraction comparison requires positive totals.")
    rng = np.random.default_rng(seed)
    first = rng.beta(
        first_successes + 0.5,
        first_total - first_successes + 0.5,
        size=samples,
    )
    second = rng.beta(
        second_successes + 0.5,
        second_total - second_successes + 0.5,
        size=samples,
    )
    difference = first - second
    point = first_successes / first_total - second_successes / second_total
    lower, upper = np.quantile(difference, (0.025, 0.975))
    return float(point), float(lower), float(upper)


def _sigma_energy_keV(energy_keV: float) -> float:
    """Return the native CeBr3 Gaussian energy-resolution sigma."""
    return max(0.5 * math.sqrt(max(0.0, energy_keV)) - 1.5, 0.5)


def _sum_peak_candidates(
    nuclide: Nuclide,
    *,
    energy_max_keV: float,
    minimum_line_probability: float,
) -> list[dict[str, object]]:
    """Return pair-sum windows, explicitly labelled as spectral candidates."""
    lines = tuple(
        line
        for line in nuclide.decay_lines
        if line.intensity >= minimum_line_probability
    )
    candidates: list[dict[str, object]] = []
    seen: set[int] = set()
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1 :]:
            energy = float(first.energy_keV + second.energy_keV)
            if energy > energy_max_keV:
                continue
            rounded_token = int(round(energy * 10.0))
            if rounded_token in seen:
                continue
            seen.add(rounded_token)
            half_width = max(4.0, 2.0 * _sigma_energy_keV(energy))
            isolated = all(
                abs(energy - line.energy_keV) > half_width
                for line in nuclide.decay_lines
            )
            candidates.append(
                {
                    "energy_keV": energy,
                    "window_half_width_keV": half_width,
                    "constituent_energies_keV": [
                        float(first.energy_keV),
                        float(second.energy_keV),
                    ],
                    "isolated_from_single_gamma_lines": isolated,
                    "semantics": (
                        "pair-energy candidate; RDM decides whether the "
                        "transition pair is physically coincident"
                    ),
                }
            )
    return sorted(candidates, key=lambda item: float(item["energy_keV"]))


def _window_count(
    spectrum: NDArray[np.float64],
    *,
    center_keV: float,
    half_width_keV: float,
) -> int:
    """Return integer pulse counts in one inclusive energy window."""
    energies = np.arange(spectrum.size, dtype=np.float64) * _NATIVE_BIN_WIDTH_KEV
    mask = np.abs(energies - center_keV) <= half_width_keV
    return int(round(float(np.sum(spectrum[mask]))))


def _case_analysis(
    *,
    design: DecayCascadeComparisonDesign,
    isotope: str,
    distance_m: float,
    rdm_spectrum: NDArray[np.float64],
    line_spectrum: NDArray[np.float64],
) -> dict[str, object]:
    """Analyze one paired distance/isotope spectrum without changing either."""
    rdm_total = int(round(float(np.sum(rdm_spectrum))))
    line_total = int(round(float(np.sum(line_spectrum))))
    if rdm_total <= 0 or line_total <= 0:
        return {
            "isotope": isotope,
            "distance_m": distance_m,
            "status": "inconclusive",
            "reason": "one or both detector spectra are empty",
            "rdm_detected_pulses": rdm_total,
            "independent_detected_pulses": line_total,
        }
    rdm_common = _rebin_common_band(
        rdm_spectrum,
        comparison_bin_width_keV=design.comparison_bin_width_keV,
    )
    line_common = _rebin_common_band(
        line_spectrum,
        comparison_bin_width_keV=design.comparison_bin_width_keV,
    )
    seed = _case_seed(
        design,
        isotope=isotope,
        distance_m=distance_m,
        emission_model="analysis_bootstrap",
    )
    tv_point, tv_lower, tv_upper = _bootstrap_tv_interval(
        rdm_common,
        line_common,
        samples=design.bootstrap_samples,
        seed=seed,
    )
    nuclide = require_nuclide(isotope)
    max_single_energy = max(line.energy_keV for line in nuclide.decay_lines)
    exclusive_threshold = max_single_energy + 3.0 * _sigma_energy_keV(
        max_single_energy
    )
    energies = np.arange(rdm_spectrum.size, dtype=np.float64) * _NATIVE_BIN_WIDTH_KEV
    exclusive_mask = energies > exclusive_threshold
    rdm_exclusive = int(round(float(np.sum(rdm_spectrum[exclusive_mask]))))
    line_exclusive = int(round(float(np.sum(line_spectrum[exclusive_mask]))))
    excess_point, excess_lower, excess_upper = _beta_difference_interval(
        rdm_exclusive,
        rdm_total,
        line_exclusive,
        line_total,
        samples=design.bootstrap_samples,
        seed=seed ^ 0xA5A5A5A5,
    )
    candidate_results: list[dict[str, object]] = []
    isolated_upper_values: list[float] = []
    for candidate_index, candidate in enumerate(
        _sum_peak_candidates(
            nuclide,
            energy_max_keV=design.energy_max_keV,
            minimum_line_probability=design.minimum_sum_line_probability,
        )
    ):
        rdm_count = _window_count(
            rdm_spectrum,
            center_keV=float(candidate["energy_keV"]),
            half_width_keV=float(candidate["window_half_width_keV"]),
        )
        line_count = _window_count(
            line_spectrum,
            center_keV=float(candidate["energy_keV"]),
            half_width_keV=float(candidate["window_half_width_keV"]),
        )
        point, lower, upper = _beta_difference_interval(
            rdm_count,
            rdm_total,
            line_count,
            line_total,
            samples=design.bootstrap_samples,
            seed=seed ^ (candidate_index + 1),
        )
        result = dict(candidate)
        result.update(
            {
                "rdm_window_count": rdm_count,
                "independent_window_count": line_count,
                "rdm_minus_independent_fraction": point,
                "fraction_difference_lower_95": lower,
                "fraction_difference_upper_95": upper,
            }
        )
        candidate_results.append(result)
        if bool(candidate["isolated_from_single_gamma_lines"]):
            isolated_upper_values.append(upper)
    maximum_sum_upper = max(isolated_upper_values, default=excess_upper)
    enough_pulses = rdm_total >= design.minimum_rdm_detected_pulses
    tv_pass = tv_upper <= design.maximum_common_band_tv
    tv_fail = tv_lower > design.maximum_common_band_tv
    coincidence_pass = max(excess_upper, maximum_sum_upper) <= (
        design.maximum_coincidence_excess_fraction
    )
    coincidence_fail = max(excess_lower, 0.0) > (
        design.maximum_coincidence_excess_fraction
    )
    if not enough_pulses:
        status = "inconclusive"
        reason = "insufficient RDM detector pulses for the predeclared gate"
    elif tv_fail or coincidence_fail:
        status = "cascade_aware_model_required"
        reason = "RDM and independent-line spectra differ beyond the gate"
    elif tv_pass and coincidence_pass:
        status = "independent_basis_adequate"
        reason = "all predeclared upper confidence bounds satisfy the gate"
    else:
        status = "inconclusive"
        reason = "confidence interval overlaps one or more gate thresholds"
    return {
        "isotope": isotope,
        "distance_m": distance_m,
        "status": status,
        "reason": reason,
        "rdm_detected_pulses": rdm_total,
        "independent_detected_pulses": line_total,
        "rdm_common_band_fraction": float(np.sum(rdm_common) / rdm_total),
        "independent_common_band_fraction": float(
            np.sum(line_common) / line_total
        ),
        "common_band_tv": tv_point,
        "common_band_tv_lower_95": tv_lower,
        "common_band_tv_upper_95": tv_upper,
        "coincidence_exclusive_threshold_keV": exclusive_threshold,
        "coincidence_exclusive_rdm_count": rdm_exclusive,
        "coincidence_exclusive_independent_count": line_exclusive,
        "coincidence_excess_fraction": excess_point,
        "coincidence_excess_lower_95": excess_lower,
        "coincidence_excess_upper_95": excess_upper,
        "maximum_isolated_sum_candidate_upper_95": maximum_sum_upper,
        "sum_peak_candidates": candidate_results,
    }


def analyze_decay_cascade_cases(
    design: DecayCascadeComparisonDesign,
    case_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Analyze all paired cases and return an immutable decision payload."""
    by_key = {
        (
            str(record["isotope"]),
            float(record["distance_m"]),
            str(record["emission_model"]),
        ): record
        for record in case_records
    }
    results: list[dict[str, object]] = []
    for isotope in design.isotopes:
        for distance in design.distances_m:
            rdm_record = by_key[
                (isotope, distance, "geant4_radioactive_decay")
            ]
            line_record = by_key[
                (isotope, distance, "independent_gamma_lines")
            ]
            rdm = np.asarray(
                np.load(str(rdm_record["spectrum_path"])),
                dtype=np.float64,
            )
            line = np.asarray(
                np.load(str(line_record["spectrum_path"])),
                dtype=np.float64,
            )
            results.append(
                _case_analysis(
                    design=design,
                    isotope=isotope,
                    distance_m=distance,
                    rdm_spectrum=rdm,
                    line_spectrum=line,
                )
            )
    statuses = {str(result["status"]) for result in results}
    if "cascade_aware_model_required" in statuses:
        overall = "cascade_aware_model_required"
    elif statuses == {"independent_basis_adequate"}:
        overall = "independent_basis_adequate"
    else:
        overall = "inconclusive"
    isotope_status: dict[str, str] = {}
    for isotope in design.isotopes:
        subset = [
            str(result["status"])
            for result in results
            if result["isotope"] == isotope
        ]
        if "cascade_aware_model_required" in subset:
            isotope_status[isotope] = "cascade_aware_model_required"
        elif set(subset) == {"independent_basis_adequate"}:
            isotope_status[isotope] = "independent_basis_adequate"
        else:
            isotope_status[isotope] = "inconclusive"
    return {
        "schema_version": _COMPARISON_SCHEMA_VERSION,
        "overall_status": overall,
        "status_by_isotope": isotope_status,
        "case_results": results,
        "automatic_runtime_model_deployment": False,
        "model_action": (
            "retain authenticated detector-cps basis"
            if overall == "independent_basis_adequate"
            else (
                "train and independently validate a cascade-aware model"
                if overall == "cascade_aware_model_required"
                else "acquire more RDM histories before deciding"
            )
        ),
    }


def _execute_comparison_cases(
    cases: Sequence[_ComparisonCase],
    *,
    max_workers: int,
) -> list[dict[str, object]]:
    """Execute cases in an ordered process pool with a fixed worker bound."""
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_run_comparison_case, cases))


def run_decay_cascade_comparison(
    *,
    design: DecayCascadeComparisonDesign,
    executable: str | Path,
    detector_model: ExportedDetectorModel,
    output_directory: str | Path,
) -> dict[str, object]:
    """Acquire all cases in bounded process-parallel Geant4 workers."""
    executable_path = Path(executable).expanduser().resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(f"Geant4 sidecar not found: {executable_path}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    case_root = output / "cases"
    case_root.mkdir()
    cases: list[_ComparisonCase] = []
    for isotope in design.isotopes:
        for distance in design.distances_m:
            distance_token = format(distance, ".3f").replace(".", "p")
            for emission_model in (
                "geant4_radioactive_decay",
                "independent_gamma_lines",
            ):
                mode_token = (
                    "rdm"
                    if emission_model == "geant4_radioactive_decay"
                    else "independent"
                )
                cases.append(
                    _ComparisonCase(
                        design=design,
                        executable=executable_path.as_posix(),
                        output_directory=(
                            case_root
                            / f"{isotope}_{distance_token}m_{mode_token}"
                        ).as_posix(),
                        detector_payload=detector_model.to_dict(),
                        isotope=isotope,
                        distance_m=distance,
                        emission_model=emission_model,
                    )
                )
    records = _execute_comparison_cases(
        cases,
        max_workers=design.case_workers,
    )
    analysis = analyze_decay_cascade_cases(design, records)
    manifest = {
        "schema_version": _COMPARISON_SCHEMA_VERSION,
        "design": design.to_dict(),
        "design_sha256": design.design_sha256,
        "nuclide_catalog_sha256": nuclide_catalog_sha256(),
        "geant4_sidecar_path": executable_path.as_posix(),
        "geant4_sidecar_sha256": _file_sha256(executable_path),
        "detector_model": detector_model.to_dict(),
        "comparison_semantics": (
            "conditional detector-pulse mark distribution; source-rate "
            "totals are intentionally not compared across incompatible "
            "detector-cps and isotropic-parent normalizations"
        ),
        "environment_semantics": (
            "empty-air distance sweep isolates decay cascade, detector "
            "response, and coincidence-window effects"
        ),
        "cases": sorted(
            records,
            key=lambda record: (
                str(record["isotope"]),
                float(record["distance_m"]),
                str(record["emission_model"]),
            ),
        ),
        "analysis": analysis,
    }
    manifest["manifest_content_sha256"] = _canonical_json_sha256(manifest)
    manifest_path = output / "decay_cascade_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def detector_model_from_runtime_config(
    payload: Mapping[str, object],
) -> ExportedDetectorModel:
    """Return the exact detector model selected by a runtime config payload."""
    raw = payload.get("detector_model", {})
    if not isinstance(raw, Mapping):
        raise TypeError("runtime detector_model must be a JSON object.")
    return ExportedDetectorModel(
        crystal_radius_m=float(raw.get("crystal_radius_m", 0.038)),
        crystal_length_m=float(raw.get("crystal_length_m", 0.076)),
        housing_thickness_m=float(raw.get("housing_thickness_m", 0.0015)),
        coincidence_window_s=float(
            raw.get(
                "coincidence_window_s",
                DEFAULT_DETECTOR_COINCIDENCE_WINDOW_S,
            )
        ),
        crystal_shape=str(raw.get("crystal_shape", "sphere")),
        crystal_material=str(raw.get("crystal_material", "cebr3")),
        housing_material=str(raw.get("housing_material", "aluminum")),
    )
