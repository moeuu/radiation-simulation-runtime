"""Fixed-quota external-Geant4 acquisition for response-mean calibration.

This module is intentionally separate from the unit-weight full-simulation
backend.  It estimates transport means with fixed source-line quotas and
persists the primary-history sufficient statistics needed to reconstruct the
sampling covariance.  Detector-response marking is integrated analytically;
no categorical detector-response draws enter the calibration artifact.

The resulting corpus is suitable for fitting the additive transport-mean
component.  It is not a substitute for station-level stochastic observations
and therefore cannot, by itself, train or authenticate likelihood-discrepancy
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.source_boundary import (
    surface_source_runtime_contract_sha256,
)
from sim.geant4_app.app import (
    Geant4AppConfig,
    Geant4Application,
    validate_mean_calibration_transport_metadata,
)
from sim.isaacsim_app.scene_builder import build_scene_description
from sim.runtime import load_runtime_config
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    AdditiveNoncollidedTransportResponse,
    fit_additive_noncollided_transport_response,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_DWELL_TIME_S,
    ACCEPTANCE_ISOTOPES,
    ACCEPTANCE_PAIR_IDS,
    acceptance_transport_seed,
)
from spectrum.geant4_acceptance_backend import (
    ACCEPTANCE_DETECTOR_POSE_XYZ,
    _FULL_SPECTRUM_RUNTIME_KEYS,
    _build_environment,
    _command_for_pair,
    _generate_sources,
    _geometry_batch,
    _request_for_command,
    _scene_payload,
    _source_payload,
)
from spectrum.mean_calibration import (
    StratifiedMeanCalibration,
    parse_mean_calibration_metadata,
)
from spectrum.native_metadata import native_source_line_token
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)


MEAN_CALIBRATION_DESIGN_SCHEMA_VERSION = 1
MEAN_CALIBRATION_PAIR_SCHEMA_VERSION = 1
MEAN_CALIBRATION_SCENE_SCHEMA_VERSION = 1
MEAN_CALIBRATION_COMPLETION_SCHEMA_VERSION = 1
MEAN_CALIBRATION_MODEL_ID = "fixed_quota_transport_mean_training_v1"
_PAIR_ARRAY_NAMES = (
    "raw_mean",
    "raw_covariance",
    "marked_mean",
    "marked_covariance",
)


def canonical_json_bytes(payload: object) -> bytes:
    """Return deterministic strict JSON bytes with one trailing newline."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    """Return the SHA-256 of the canonical JSON encoding."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _npz_bytes(array: NDArray[np.float64]) -> bytes:
    """Return deterministic compressed NumPy bytes for one float64 array."""
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        array=np.ascontiguousarray(array, dtype=np.float64),
    )
    return buffer.getvalue()


def _write_immutable_bytes(path: Path, content: bytes) -> Path:
    """Atomically create one immutable file or verify an exact resumption."""
    destination = path.resolve()
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError(
                f"Refusing to overwrite incompatible artifact: {destination}."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_bytes(content)
    os.replace(temporary, destination)
    return destination


def _strict_positive_integer(value: object, *, field_name: str) -> int:
    """Return one positive JSON integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return int(value)


def _strict_boolean(value: object, *, field_name: str) -> bool:
    """Return one JSON boolean without truth-value coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a JSON boolean.")
    return value


def _strict_unique_integer_sequence(
    value: object,
    *,
    field_name: str,
    minimum: int,
    maximum: int | None = None,
) -> tuple[int, ...]:
    """Return one nonempty ordered unique JSON-integer sequence."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"{field_name} must be a nonempty sequence.")
    result: list[int] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < minimum
            or (maximum is not None and item > maximum)
        ):
            raise ValueError(
                f"{field_name} contains an out-of-range JSON integer."
            )
        result.append(int(item))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return tuple(result)


def _strict_scenario_sequence(value: object) -> tuple[str, ...]:
    """Return one nonempty unique sequence of implemented scenarios."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(item, str)
            or item not in VALIDATION_SCENARIO_IDS
            for item in value
        )
    ):
        raise ValueError(
            "Mean-calibration design is not the exact predeclared contract: "
            "scenario_ids must contain implemented scenario identifiers."
        )
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("scenario_ids must not contain duplicates.")
    return result


def build_predeclared_mean_calibration_design(
    *,
    histories_per_source_line: int,
    angle_strata_mu: int,
    angle_strata_phi: int,
    forced_collision: bool = False,
    training_scene_seeds: Sequence[int] = DESIGNATED_TRAINING_SCENE_SEEDS,
    scenario_ids: Sequence[str] = VALIDATION_SCENARIO_IDS,
    shield_pair_ids: Sequence[int] = ACCEPTANCE_PAIR_IDS,
) -> dict[str, object]:
    """Build the sole predeclared training-only fixed-quota design.

    The design deliberately excludes independent holdout data from model
    construction.  An all-64 holdout remains optional release evidence and is
    never a startup dependency for a model that already has an authenticated
    training contract.
    """
    quota = _strict_positive_integer(
        histories_per_source_line,
        field_name="histories_per_source_line",
    )
    mu_count = _strict_positive_integer(
        angle_strata_mu,
        field_name="angle_strata_mu",
    )
    phi_count = _strict_positive_integer(
        angle_strata_phi,
        field_name="angle_strata_phi",
    )
    use_forced_collision = _strict_boolean(
        forced_collision,
        field_name="forced_collision",
    )
    scene_seeds = _strict_unique_integer_sequence(
        training_scene_seeds,
        field_name="training_scene_seeds",
        minimum=0,
    )
    scenarios = _strict_scenario_sequence(scenario_ids)
    pair_ids = _strict_unique_integer_sequence(
        shield_pair_ids,
        field_name="shield_pair_ids",
        minimum=0,
        maximum=63,
    )
    stratum_count = mu_count * phi_count
    if quota % stratum_count != 0 or quota // stratum_count < 2:
        raise ValueError(
            "histories_per_source_line must allocate at least two histories "
            "to every mu/phi stratum."
        )
    return {
        "schema_version": MEAN_CALIBRATION_DESIGN_SCHEMA_VERSION,
        "model": MEAN_CALIBRATION_MODEL_ID,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(scene_seeds),
        "scenario_ids": list(scenarios),
        "shield_pair_ids": list(pair_ids),
        "histories_per_source_line": quota,
        "angle_strata_mu": mu_count,
        "angle_strata_phi": phi_count,
        "angle_stratum_count": stratum_count,
        "forced_collision": use_forced_collision,
        "source_rate_model": "detector_cps_1m",
        "detector_scoring_mode": "incident_gamma_energy",
        "secondary_transport_mode": "full_transport",
        "detector_response_mode": "rao_blackwell_analytic_operator",
        "detector_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "background_cps": 0.0,
        "dead_time_tau_s": 0.0,
        "holdout_consumed_by_training": False,
        "all64_holdout_role": "optional_independent_release_evidence",
    }


def validate_predeclared_mean_calibration_design(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate and canonicalize one fixed training design."""
    if not isinstance(payload, Mapping):
        raise TypeError("Mean-calibration design must be a mapping.")
    expected = build_predeclared_mean_calibration_design(
        histories_per_source_line=_strict_positive_integer(
            payload.get("histories_per_source_line"),
            field_name="histories_per_source_line",
        ),
        angle_strata_mu=_strict_positive_integer(
            payload.get("angle_strata_mu"),
            field_name="angle_strata_mu",
        ),
        angle_strata_phi=_strict_positive_integer(
            payload.get("angle_strata_phi"),
            field_name="angle_strata_phi",
        ),
        forced_collision=_strict_boolean(
            payload.get("forced_collision"),
            field_name="forced_collision",
        ),
        training_scene_seeds=_strict_unique_integer_sequence(
            payload.get("training_scene_seeds"),
            field_name="training_scene_seeds",
            minimum=0,
        ),
        scenario_ids=_strict_scenario_sequence(
            payload.get("scenario_ids")
        ),
        shield_pair_ids=_strict_unique_integer_sequence(
            payload.get("shield_pair_ids"),
            field_name="shield_pair_ids",
            minimum=0,
            maximum=63,
        ),
    )
    if dict(payload) != expected:
        raise ValueError(
            "Mean-calibration design is not the exact predeclared contract."
        )
    return expected


def build_mean_calibration_app_payload(
    runtime_config: Mapping[str, object],
    *,
    repository_root: str | Path,
    design: Mapping[str, object],
) -> dict[str, object]:
    """Return an isolated external-Geant4 fixed-quota application payload."""
    resolved_design = validate_predeclared_mean_calibration_design(design)
    payload = {
        key: value
        for key, value in dict(runtime_config).items()
        if key not in _FULL_SPECTRUM_RUNTIME_KEYS
    }
    executable_raw = payload.get("executable_path", "build/geant4_sidecar")
    if not isinstance(executable_raw, str) or not executable_raw:
        raise ValueError("executable_path must be a nonempty string.")
    executable = Path(executable_raw)
    if not executable.is_absolute():
        executable = Path(repository_root).resolve() / executable
    payload.update(
        {
            "executable_path": executable.resolve().as_posix(),
            "engine_mode": "external",
            "source_rate_model": "detector_cps_1m",
            "detector_scoring_mode": "incident_gamma_energy",
            "secondary_transport_mode": "full_transport",
            "sample_detector_response": False,
            "validation_entry_class_spectra": True,
            "background_cps": 0.0,
            "dead_time_tau_s": 0.0,
            "primary_sampling_fraction": 1.0,
            "target_sampled_primaries": None,
            "accelerated_weighted_transport_enable": False,
            "mean_calibration_histories_per_source_line": int(
                resolved_design["histories_per_source_line"]
            ),
            "mean_calibration_angle_strata_mu": int(
                resolved_design["angle_strata_mu"]
            ),
            "mean_calibration_angle_strata_phi": int(
                resolved_design["angle_strata_phi"]
            ),
            "mean_calibration_forced_collision": bool(
                resolved_design["forced_collision"]
            ),
        }
    )
    config = Geant4AppConfig.from_dict(payload)
    if (
        config.engine_mode != "external"
        or config.source_rate_model != "detector_cps_1m"
        or config.detector_scoring_mode != "incident_gamma_energy"
        or config.secondary_transport_mode != "full_transport"
        or config.sample_detector_response
        or not config.validation_entry_class_spectra
        or config.background_cps != 0.0
        or config.dead_time_tau_s != 0.0
        or config.primary_sampling_fraction != 1.0
        or config.target_sampled_primaries is not None
        or config.accelerated_weighted_transport_enable
        or (
            config.mean_calibration_forced_collision
            != bool(resolved_design["forced_collision"])
        )
    ):
        raise RuntimeError("Mean-calibration application contract drifted.")
    return payload


@dataclass(frozen=True)
class MeanCalibrationLayout:
    """Resolve immutable fixed-quota training artifact paths."""

    root: Path

    @property
    def design_path(self) -> Path:
        """Return the immutable predeclared design path."""
        return self.root / "training_design.json"

    @property
    def completion_path(self) -> Path:
        """Return the complete-training manifest path."""
        return self.root / "training_complete.json"

    @property
    def additive_model_path(self) -> Path:
        """Return the fitted additive-scatter model path."""
        return self.root / "additive_scatter_model.json"

    def pair_directory(
        self,
        *,
        scene_seed: int,
        scenario_id: str,
        shield_pair_id: int,
    ) -> Path:
        """Return one pair artifact directory."""
        return (
            self.root
            / "pairs"
            / f"scene_{int(scene_seed)}"
            / str(scenario_id)
            / f"pair_{int(shield_pair_id):02d}"
        )

    def scene_manifest_path(self, scene_seed: int) -> Path:
        """Return one complete-scene corpus manifest path."""
        return self.root / "scenes" / f"training_scene_{int(scene_seed)}.json"


def initialize_mean_calibration_layout(
    *,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
) -> Path:
    """Create or verify the immutable predeclared training design."""
    payload = validate_predeclared_mean_calibration_design(design)
    return _write_immutable_bytes(
        layout.design_path,
        canonical_json_bytes(payload),
    )


def _class_totals_payload(
    calibration: StratifiedMeanCalibration,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return JSON-compatible source-line entry-class moments."""
    return {
        line_token: {
            entry_class: {
                "mean_count": float(values[0]),
                "sampling_variance": float(values[1]),
            }
            for entry_class, values in classes.items()
        }
        for line_token, classes in (
            calibration.entry_class_line_totals().items()
        )
    }


def write_mean_calibration_pair_artifact(
    *,
    directory: str | Path,
    provenance: Mapping[str, object],
    calibration: StratifiedMeanCalibration,
    native_spectrum: Sequence[float],
    native_spectrum_variance: Sequence[float],
    response_operator_br: NDArray[np.float64],
    geometry: Mapping[str, object],
) -> Path:
    """Persist one immutable pair manifest, moments and sufficient statistics."""
    target = Path(directory).resolve()
    calibration.validate_native_arrays(
        native_spectrum,
        native_spectrum_variance,
    )
    arrays = {
        "raw_mean": calibration.raw_mean(),
        "raw_covariance": calibration.raw_covariance(),
        "marked_mean": calibration.marked_mean(response_operator_br),
        "marked_covariance": calibration.marked_covariance(
            response_operator_br
        ),
    }
    array_files: dict[str, dict[str, object]] = {}
    for name in _PAIR_ARRAY_NAMES:
        encoded = _npz_bytes(arrays[name])
        path = target / f"{name}.npz"
        _write_immutable_bytes(path, encoded)
        array_files[name] = {
            "path": path.name,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "shape": list(arrays[name].shape),
            "dtype": "float64",
            "format": "numpy_npz_deflate_v1",
            "member": "array",
        }
    payload = {
        "schema_version": MEAN_CALIBRATION_PAIR_SCHEMA_VERSION,
        "model": "fixed_quota_transport_mean_pair_v1",
        "provenance": dict(provenance),
        "geometry": dict(geometry),
        "detector_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "detector_response_application": (
            "analytic_conditional_expectation_no_categorical_sampling"
        ),
        "sampling_covariance_scope": (
            "fixed_quota_transport_mean_not_station_count_dispersion"
        ),
        "entry_class_line_moments": _class_totals_payload(calibration),
        "sufficient_statistics": calibration.to_payload(),
        "arrays": array_files,
    }
    manifest_path = target / "manifest.json"
    _write_immutable_bytes(manifest_path, canonical_json_bytes(payload))
    load_mean_calibration_pair_artifact(manifest_path)
    return manifest_path


def load_mean_calibration_pair_artifact(
    manifest_path: str | Path,
) -> tuple[dict[str, object], dict[str, NDArray[np.float64]]]:
    """Load and authenticate one immutable pair artifact."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != MEAN_CALIBRATION_PAIR_SCHEMA_VERSION
        or payload.get("model")
        != "fixed_quota_transport_mean_pair_v1"
        or payload.get("detector_response_contract_sha256")
        != NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        or payload.get("detector_response_application")
        != "analytic_conditional_expectation_no_categorical_sampling"
        or payload.get("sampling_covariance_scope")
        != "fixed_quota_transport_mean_not_station_count_dispersion"
        or not isinstance(payload.get("arrays"), Mapping)
    ):
        raise ValueError("Mean-calibration pair manifest is invalid.")
    arrays_payload = payload["arrays"]
    if set(arrays_payload) != set(_PAIR_ARRAY_NAMES):
        raise ValueError("Mean-calibration pair array set is invalid.")
    arrays: dict[str, NDArray[np.float64]] = {}
    for name in _PAIR_ARRAY_NAMES:
        item = arrays_payload[name]
        if not isinstance(item, Mapping):
            raise TypeError("Mean-calibration array descriptor is invalid.")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).name != relative
        ):
            raise ValueError("Mean-calibration array path is invalid.")
        array_path = path.parent / relative
        if file_sha256(array_path) != item.get("sha256"):
            raise ValueError("Mean-calibration array hash mismatch.")
        if (
            item.get("format") != "numpy_npz_deflate_v1"
            or item.get("member") != "array"
        ):
            raise ValueError("Mean-calibration array encoding is invalid.")
        with np.load(array_path, allow_pickle=False) as archive:
            if archive.files != ["array"]:
                raise ValueError(
                    "Mean-calibration array archive is noncanonical."
                )
            array = np.asarray(archive["array"])
        if (
            array.dtype != np.float64
            or list(array.shape) != item.get("shape")
            or np.any(~np.isfinite(array))
        ):
            raise ValueError("Mean-calibration array contents are invalid.")
        arrays[name] = np.asarray(array, dtype=np.float64)
    bin_count = arrays["raw_mean"].size
    if (
        arrays["raw_mean"].shape != (bin_count,)
        or arrays["marked_mean"].shape != (bin_count,)
        or arrays["raw_covariance"].shape != (bin_count, bin_count)
        or arrays["marked_covariance"].shape != (bin_count, bin_count)
        or np.any(arrays["raw_mean"] < 0.0)
        or np.any(arrays["marked_mean"] < 0.0)
        or not np.allclose(
            arrays["raw_covariance"],
            arrays["raw_covariance"].T,
            rtol=0.0,
            atol=1.0e-10,
        )
        or not np.allclose(
            arrays["marked_covariance"],
            arrays["marked_covariance"].T,
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise ValueError("Mean-calibration moments are inconsistent.")
    return payload, arrays


def _metadata_variance(metadata: Mapping[str, Any]) -> NDArray[np.float64]:
    """Return the native fixed-quota spectrum-variance vector."""
    value = metadata.get("spectrum_count_variance")
    if isinstance(value, str):
        raw: object = value.split(",")
    else:
        raw = value
    try:
        variance = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native calibration response is missing spectrum variance."
        ) from exc
    if (
        variance.shape != (NATIVE_GEANT4_BIN_COUNT,)
        or np.any(~np.isfinite(variance))
        or np.any(variance < 0.0)
    ):
        raise RuntimeError("Native calibration spectrum variance is invalid.")
    return variance


class ExternalGeant4MeanCalibrationBackend:
    """Acquire fixed-quota transport means without altering standard runtime."""

    backend_id = "external_geant4_fixed_quota_mean_v1"

    def __init__(
        self,
        *,
        runtime_config_path: str | Path,
        repository_root: str | Path,
        design: Mapping[str, object],
    ) -> None:
        """Load the standard geometry config and isolate calibration overrides."""
        self.repository_root = Path(repository_root).resolve()
        self.runtime_config_path = Path(runtime_config_path).resolve()
        self.runtime_config = load_runtime_config(self.runtime_config_path)
        self.design = validate_predeclared_mean_calibration_design(design)
        self.app_payload = build_mean_calibration_app_payload(
            self.runtime_config,
            repository_root=self.repository_root,
            design=self.design,
        )
        self.app_config = Geant4AppConfig.from_dict(self.app_payload)
        executable = Path(str(self.app_config.executable_path)).resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"Native Geant4 executable is missing: {executable}."
            )
        self.runtime_config_sha256 = file_sha256(self.runtime_config_path)
        self.native_executable_sha256 = file_sha256(executable)
        self.model = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=0.0,
            background_rate_cps=0.0,
        )

    def _kernel(self, grid: object) -> object:
        """Build the shared batched continuous-geometry response kernel."""
        observation_payload = {
            key: value
            for key, value in self.runtime_config.items()
            if key not in _FULL_SPECTRUM_RUNTIME_KEYS
        }
        observation = build_runtime_observation_model(
            observation_payload,
            isotopes=ACCEPTANCE_ISOTOPES,
        )
        if observation.additive_scatter_response is not None:
            raise RuntimeError(
                "Mean calibration geometry must precede scatter fitting."
            )
        kernel = continuous_kernel_from_observation_model(
            observation,
            obstacle_grid=grid,
            use_gpu=bool(observation_payload.get("use_gpu", False)),
        )
        kernel.gpu_device = str(
            observation_payload.get("gpu_device", "cuda")
        )
        kernel.gpu_dtype = str(
            observation_payload.get("gpu_dtype", "float64")
        )
        return kernel

    def acquire_scenario(
        self,
        *,
        layout: MeanCalibrationLayout,
        scene_seed: int,
        scenario_id: str,
        pair_ids: Sequence[int] | None = None,
    ) -> tuple[Path, ...]:
        """Acquire selected pairs for one predeclared training scenario."""
        selected_pair_ids = (
            tuple(int(value) for value in self.design["shield_pair_ids"])
            if pair_ids is None
            else tuple(int(value) for value in pair_ids)
        )
        if (
            scene_seed not in self.design["training_scene_seeds"]
            or scenario_id not in self.design["scenario_ids"]
            or not selected_pair_ids
            or len(set(selected_pair_ids)) != len(selected_pair_ids)
            or any(
                pair_id not in self.design["shield_pair_ids"]
                for pair_id in selected_pair_ids
            )
        ):
            raise ValueError("Acquisition request is outside the training design.")
        environment, grid, instances = _build_environment(
            scene_seed=scene_seed,
            obstacle_height_m=float(self.app_config.obstacle_height_m),
            author_room_boundaries=bool(
                self.app_config.author_room_boundary_prims
            ),
            room_boundary_thickness_m=float(
                self.runtime_config.get("room_boundary_thickness_m", 0.1)
            ),
        )
        sources = _generate_sources(
            environment=environment,
            grid=grid,
            scene_seed=scene_seed,
            scenario_id=scenario_id,
            obstacle_height_m=float(self.app_config.obstacle_height_m),
        )
        if not sources:
            return tuple(
                self._write_exact_zero_pair(
                    layout=layout,
                    scene_seed=scene_seed,
                    scenario_id=scenario_id,
                    shield_pair_id=int(pair_id),
                )
                for pair_id in selected_pair_ids
            )
        kernel = self._kernel(grid)
        geometry = _geometry_batch(
            kernel=kernel,
            model=self.model,
            detector_pose_xyz=ACCEPTANCE_DETECTOR_POSE_XYZ,
            sources=sources,
        )
        app = Geant4Application(app_config=dict(self.app_payload))
        app.reset(
            build_scene_description(
                _scene_payload(
                    grid=grid,
                    instances=instances,
                    sources=sources,
                    author_room_boundaries=bool(
                        self.app_config.author_room_boundary_prims
                    ),
                    obstacle_material=str(
                        self.runtime_config.get(
                            "obstacle_material",
                            "concrete",
                        )
                    ),
                )
            )
        )
        exported = getattr(app.engine, "scene", None)
        if exported is None:
            app.close()
            raise RuntimeError("Mean-calibration native scene was not exported.")
        source_payloads = tuple(_source_payload(source) for source in sources)
        source_hash = surface_source_runtime_contract_sha256(source_payloads)
        paths: list[Path] = []
        try:
            for pair_id in selected_pair_ids:
                command = _command_for_pair(int(pair_id))
                transport_seed = acceptance_transport_seed(
                    scene_seed=scene_seed,
                    scenario_id=scenario_id,
                    shield_pair_id=int(pair_id),
                )
                request = _request_for_command(
                    app,
                    command,
                    seed=transport_seed,
                    dwell_time_s=ACCEPTANCE_DWELL_TIME_S,
                )
                spectrum, raw_metadata = app.engine.simulate(request)
                metadata = dict(raw_metadata)
                validate_mean_calibration_transport_metadata(
                    metadata,
                    expected_histories_per_source_line=int(
                        self.design["histories_per_source_line"]
                    ),
                    expected_angle_strata_mu=int(
                        self.design["angle_strata_mu"]
                    ),
                    expected_angle_strata_phi=int(
                        self.design["angle_strata_phi"]
                    ),
                    expected_forced_collision=bool(
                        self.design["forced_collision"]
                    ),
                    expected_source_rate_model="detector_cps_1m",
                    expected_thread_count=self.app_config.thread_count,
                    expected_physics_profile=self.app_config.physics_profile,
                    expected_detector_scoring_mode=(
                        "incident_gamma_energy"
                    ),
                    expected_secondary_transport_mode="full_transport",
                    expected_source_bias_mode=self.app_config.source_bias_mode,
                )
                calibration = parse_mean_calibration_metadata(
                    metadata,
                    bin_count=NATIVE_GEANT4_BIN_COUNT,
                )
                native_variance = _metadata_variance(metadata)
                calibration.validate_native_arrays(
                    spectrum,
                    native_variance,
                )
                pair_index = int(pair_id)
                pair_geometry = {
                    "unattenuated_source_line_rate_sl": (
                        geometry.unattenuated_vsl[pair_index].tolist()
                    ),
                    "uncollided_source_line_rate_sl": (
                        geometry.uncollided_vsl[pair_index].tolist()
                    ),
                    "transport_features_slf": (
                        geometry.features_vslf[pair_index].tolist()
                    ),
                    "additive_scatter_basis_slf": (
                        geometry.scatter_basis_vslf[pair_index].tolist()
                    ),
                }
                provenance = self._pair_provenance(
                    scene_seed=scene_seed,
                    scenario_id=scenario_id,
                    shield_pair_id=pair_index,
                    transport_seed=transport_seed,
                    scene_hash=str(exported.scene_hash),
                    source_hash=source_hash,
                    source_payloads=source_payloads,
                )
                paths.append(
                    write_mean_calibration_pair_artifact(
                        directory=layout.pair_directory(
                            scene_seed=scene_seed,
                            scenario_id=scenario_id,
                            shield_pair_id=pair_index,
                        ),
                        provenance=provenance,
                        calibration=calibration,
                        native_spectrum=spectrum,
                        native_spectrum_variance=native_variance,
                        response_operator_br=self.model.response_operator_br,
                        geometry=pair_geometry,
                    )
                )
        finally:
            app.close()
        return tuple(paths)

    def _pair_provenance(
        self,
        *,
        scene_seed: int,
        scenario_id: str,
        shield_pair_id: int,
        transport_seed: int,
        scene_hash: str,
        source_hash: str,
        source_payloads: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Return immutable acquisition identity for one pair."""
        return {
            "backend_id": self.backend_id,
            "design_sha256": canonical_json_sha256(self.design),
            "runtime_config_sha256": self.runtime_config_sha256,
            "native_executable_sha256": self.native_executable_sha256,
            "scene_seed": int(scene_seed),
            "scenario_id": str(scenario_id),
            "shield_pair_id": int(shield_pair_id),
            "transport_seed": int(transport_seed),
            "dwell_time_s": ACCEPTANCE_DWELL_TIME_S,
            "scene_hash": str(scene_hash),
            "surface_source_contract_sha256": str(source_hash),
            "sources": [dict(value) for value in source_payloads],
            "holdout_artifacts_consumed": False,
        }

    def _write_exact_zero_pair(
        self,
        *,
        layout: MeanCalibrationLayout,
        scene_seed: int,
        scenario_id: str,
        shield_pair_id: int,
    ) -> Path:
        """Write the exact source-free, zero-background calibration identity."""
        directory = layout.pair_directory(
            scene_seed=scene_seed,
            scenario_id=scenario_id,
            shield_pair_id=shield_pair_id,
        )
        zero_vector = np.zeros(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
        zero_matrix = np.zeros(
            (NATIVE_GEANT4_BIN_COUNT, NATIVE_GEANT4_BIN_COUNT),
            dtype=np.float64,
        )
        arrays = {
            "raw_mean": zero_vector,
            "raw_covariance": zero_matrix,
            "marked_mean": zero_vector,
            "marked_covariance": zero_matrix,
        }
        array_files: dict[str, dict[str, object]] = {}
        for name in _PAIR_ARRAY_NAMES:
            encoded = _npz_bytes(arrays[name])
            path = directory / f"{name}.npz"
            _write_immutable_bytes(path, encoded)
            array_files[name] = {
                "path": path.name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "shape": list(arrays[name].shape),
                "dtype": "float64",
                "format": "numpy_npz_deflate_v1",
                "member": "array",
            }
        payload = {
            "schema_version": MEAN_CALIBRATION_PAIR_SCHEMA_VERSION,
            "model": "fixed_quota_transport_mean_pair_v1",
            "provenance": {
                "backend_id": self.backend_id,
                "design_sha256": canonical_json_sha256(self.design),
                "runtime_config_sha256": self.runtime_config_sha256,
                "native_executable_sha256": (
                    self.native_executable_sha256
                ),
                "scene_seed": int(scene_seed),
                "scenario_id": str(scenario_id),
                "shield_pair_id": int(shield_pair_id),
                "transport_seed": None,
                "dwell_time_s": ACCEPTANCE_DWELL_TIME_S,
                "scene_hash": None,
                "surface_source_contract_sha256": None,
                "sources": [],
                "holdout_artifacts_consumed": False,
                "exact_zero_reason": (
                    "no_sources_background_zero_dead_time_zero"
                ),
            },
            "geometry": {
                "unattenuated_source_line_rate_sl": [],
                "uncollided_source_line_rate_sl": [],
                "transport_features_slf": [],
                "additive_scatter_basis_slf": [],
            },
            "detector_response_contract_sha256": (
                NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
            ),
            "detector_response_application": (
                "analytic_conditional_expectation_no_categorical_sampling"
            ),
            "sampling_covariance_scope": (
                "fixed_quota_transport_mean_not_station_count_dispersion"
            ),
            "entry_class_line_moments": {},
            "sufficient_statistics": None,
            "arrays": array_files,
        }
        manifest = directory / "manifest.json"
        _write_immutable_bytes(manifest, canonical_json_bytes(payload))
        load_mean_calibration_pair_artifact(manifest)
        return manifest


def build_mean_calibration_scene_manifest(
    *,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
    scene_seed: int,
) -> dict[str, object]:
    """Authenticate every predeclared scenario/pair artifact for one scene."""
    resolved = validate_predeclared_mean_calibration_design(design)
    scene_seeds = tuple(int(value) for value in resolved["training_scene_seeds"])
    scenario_ids = tuple(str(value) for value in resolved["scenario_ids"])
    pair_ids = tuple(int(value) for value in resolved["shield_pair_ids"])
    if scene_seed not in scene_seeds:
        raise ValueError("scene_seed is outside the training design.")
    pair_hashes: dict[str, dict[str, str]] = {}
    for scenario_id in scenario_ids:
        scenario_hashes: dict[str, str] = {}
        for pair_id in pair_ids:
            manifest = (
                layout.pair_directory(
                    scene_seed=scene_seed,
                    scenario_id=scenario_id,
                    shield_pair_id=pair_id,
                )
                / "manifest.json"
            )
            payload, _ = load_mean_calibration_pair_artifact(manifest)
            provenance = payload.get("provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("design_sha256")
                != canonical_json_sha256(resolved)
                or provenance.get("scene_seed") != scene_seed
                or provenance.get("scenario_id") != scenario_id
                or provenance.get("shield_pair_id") != pair_id
                or provenance.get("holdout_artifacts_consumed") is not False
            ):
                raise ValueError("Mean-calibration pair identity is stale.")
            scenario_hashes[str(pair_id)] = file_sha256(manifest)
        pair_hashes[scenario_id] = scenario_hashes
    return {
        "schema_version": MEAN_CALIBRATION_SCENE_SCHEMA_VERSION,
        "model": "fixed_quota_transport_mean_scene_v1",
        "design_sha256": canonical_json_sha256(resolved),
        "scene_seed": int(scene_seed),
        "scenario_ids": list(scenario_ids),
        "shield_pair_ids": list(pair_ids),
        "pair_manifest_sha256_by_scenario": pair_hashes,
        "complete": True,
    }


def freeze_mean_calibration_scene_manifest(
    *,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
    scene_seed: int,
) -> Path:
    """Write or verify one complete immutable scene manifest."""
    payload = build_mean_calibration_scene_manifest(
        layout=layout,
        design=design,
        scene_seed=scene_seed,
    )
    return _write_immutable_bytes(
        layout.scene_manifest_path(scene_seed),
        canonical_json_bytes(payload),
    )


def build_mean_calibration_completion_manifest(
    *,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate every designated training scene without reading holdout."""
    resolved = validate_predeclared_mean_calibration_design(design)
    scene_seeds = tuple(int(value) for value in resolved["training_scene_seeds"])
    scene_hashes: dict[str, str] = {}
    for scene_seed in scene_seeds:
        expected = build_mean_calibration_scene_manifest(
            layout=layout,
            design=resolved,
            scene_seed=scene_seed,
        )
        path = layout.scene_manifest_path(scene_seed)
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("Mean-calibration scene manifest is stale.")
        scene_hashes[str(scene_seed)] = file_sha256(path)
    return {
        "schema_version": MEAN_CALIBRATION_COMPLETION_SCHEMA_VERSION,
        "model": "fixed_quota_transport_mean_training_complete_v1",
        "design_sha256": canonical_json_sha256(resolved),
        "training_scene_seeds": list(scene_seeds),
        "scenario_ids": list(resolved["scenario_ids"]),
        "shield_pair_ids": list(resolved["shield_pair_ids"]),
        "scene_manifest_sha256_by_seed": scene_hashes,
        "holdout_artifacts_consumed": False,
        "complete": True,
    }


def freeze_mean_calibration_completion_manifest(
    *,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
) -> Path:
    """Write the complete training-only corpus manifest."""
    payload = build_mean_calibration_completion_manifest(
        layout=layout,
        design=design,
    )
    return _write_immutable_bytes(
        layout.completion_path,
        canonical_json_bytes(payload),
    )


@dataclass(frozen=True)
class MeanCalibrationTrainingRow:
    """Store one physical additive-scatter regression row."""

    scene_id: str
    feature_basis: tuple[float, ...]
    scatter_fraction: float
    sample_weight: float


def fit_additive_scatter_training_rows(
    rows: Sequence[MeanCalibrationTrainingRow],
    *,
    scene_manifest_sha256_by_seed: Mapping[str, str],
    training_scene_seeds: Sequence[int] = DESIGNATED_TRAINING_SCENE_SEEDS,
    scenario_ids: Sequence[str] = VALIDATION_SCENARIO_IDS,
    shield_pair_ids: Sequence[int] = ACCEPTANCE_PAIR_IDS,
) -> AdditiveNoncollidedTransportResponse:
    """Fit the existing physical additive model from training-only rows."""
    declared_seeds = _strict_unique_integer_sequence(
        training_scene_seeds,
        field_name="training_scene_seeds",
        minimum=0,
    )
    declared_scenarios = _strict_scenario_sequence(scenario_ids)
    declared_pair_ids = _strict_unique_integer_sequence(
        shield_pair_ids,
        field_name="shield_pair_ids",
        minimum=0,
        maximum=63,
    )
    expected_seed_keys = {str(value) for value in declared_seeds}
    if (
        not rows
        or set(scene_manifest_sha256_by_seed) != expected_seed_keys
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in scene_manifest_sha256_by_seed.values()
        )
    ):
        raise ValueError("Complete authenticated training scenes are required.")
    features = np.asarray(
        [row.feature_basis for row in rows],
        dtype=np.float64,
    )
    targets = np.asarray(
        [row.scatter_fraction for row in rows],
        dtype=np.float64,
    )
    weights = np.asarray(
        [row.sample_weight for row in rows],
        dtype=np.float64,
    )
    scene_ids = [row.scene_id for row in rows]
    training_manifest = {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(declared_seeds),
        "scenario_ids": list(declared_scenarios),
        "pair_ids_by_scene": {
            str(seed): list(declared_pair_ids)
            for seed in declared_seeds
        },
        "artifact_sha256_by_scene": dict(
            scene_manifest_sha256_by_seed
        ),
        "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
        "selection_objective": (
            "leave_one_training_scene_out_weighted_log1p_mse"
        ),
    }
    return fit_additive_noncollided_transport_response(
        features,
        targets,
        weights,
        scene_ids,
        training_manifest=training_manifest,
    )


def _training_rows_from_pair(
    payload: Mapping[str, object],
    *,
    model: GeometryConditionedSpectralModel,
) -> tuple[MeanCalibrationTrainingRow, ...]:
    """Extract source-line additive-scatter rows from one pair artifact."""
    provenance = payload.get("provenance")
    geometry = payload.get("geometry")
    moments = payload.get("entry_class_line_moments")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(geometry, Mapping)
        or not isinstance(moments, Mapping)
    ):
        raise TypeError("Mean-calibration pair training fields are invalid.")
    sources = provenance.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise TypeError("Mean-calibration source rows are invalid.")
    if not sources:
        return ()
    unattenuated = np.asarray(
        geometry.get("unattenuated_source_line_rate_sl"),
        dtype=np.float64,
    )
    basis = np.asarray(
        geometry.get("additive_scatter_basis_slf"),
        dtype=np.float64,
    )
    line_rows = model.line_identity
    if (
        unattenuated.shape != (len(sources), len(line_rows))
        or basis.shape
        != (
            len(sources),
            len(line_rows),
            len(ADDITIVE_SCATTER_FEATURE_ORDER),
        )
        or np.any(~np.isfinite(unattenuated))
        or np.any(unattenuated < 0.0)
        or np.any(~np.isfinite(basis))
        or np.any(basis < 0.0)
    ):
        raise ValueError("Mean-calibration geometry arrays are invalid.")
    scene_id = str(provenance.get("scene_seed"))
    result: list[MeanCalibrationTrainingRow] = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError("Mean-calibration source row must be a mapping.")
        isotope = str(source.get("isotope"))
        for line_index, line in enumerate(line_rows):
            if line["isotope"] != isotope:
                continue
            token = native_source_line_token(
                source_index=source_index,
                isotope=isotope,
                energy_keV=float(line["energy_keV"]),
            )
            class_moments = moments.get(token)
            if not isinstance(class_moments, Mapping):
                raise ValueError(
                    f"Missing fixed-quota entry moments for {token}."
                )
            interacted = class_moments.get("interacted_primary")
            secondary = class_moments.get("secondary")
            if not isinstance(interacted, Mapping) or not isinstance(
                secondary,
                Mapping,
            ):
                raise TypeError("Entry-class moments are invalid.")
            scatter_count = float(interacted["mean_count"]) + float(
                secondary["mean_count"]
            )
            denominator = (
                float(unattenuated[source_index, line_index])
                * float(provenance["dwell_time_s"])
            )
            if denominator <= 0.0:
                continue
            result.append(
                MeanCalibrationTrainingRow(
                    scene_id=scene_id,
                    feature_basis=tuple(
                        float(value)
                        for value in basis[source_index, line_index]
                    ),
                    scatter_fraction=max(scatter_count, 0.0) / denominator,
                    sample_weight=denominator,
                )
            )
    return tuple(result)


def fit_additive_scatter_from_complete_mean_calibration(
    *,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
    model: GeometryConditionedSpectralModel,
) -> AdditiveNoncollidedTransportResponse:
    """Fit and freeze additive scatter after authenticating the full design."""
    resolved = validate_predeclared_mean_calibration_design(design)
    expected_completion = build_mean_calibration_completion_manifest(
        layout=layout,
        design=resolved,
    )
    observed_completion = json.loads(
        layout.completion_path.read_text(encoding="utf-8")
    )
    if observed_completion != expected_completion:
        raise ValueError("Mean-calibration completion manifest is stale.")
    rows: list[MeanCalibrationTrainingRow] = []
    scene_seeds = tuple(int(value) for value in resolved["training_scene_seeds"])
    scenario_ids = tuple(str(value) for value in resolved["scenario_ids"])
    pair_ids = tuple(int(value) for value in resolved["shield_pair_ids"])
    for scene_seed in scene_seeds:
        for scenario_id in scenario_ids:
            for pair_id in pair_ids:
                manifest = (
                    layout.pair_directory(
                        scene_seed=scene_seed,
                        scenario_id=scenario_id,
                        shield_pair_id=pair_id,
                    )
                    / "manifest.json"
                )
                payload, _ = load_mean_calibration_pair_artifact(manifest)
                rows.extend(_training_rows_from_pair(payload, model=model))
    response = fit_additive_scatter_training_rows(
        rows,
        scene_manifest_sha256_by_seed=expected_completion[
            "scene_manifest_sha256_by_seed"
        ],
        training_scene_seeds=scene_seeds,
        scenario_ids=scenario_ids,
        shield_pair_ids=pair_ids,
    )
    _write_immutable_bytes(
        layout.additive_model_path,
        canonical_json_bytes(response.to_payload()),
    )
    return response


def freeze_runtime_ready_model(
    *,
    output_path: str | Path,
    model: GeometryConditionedSpectralModel,
    additive_response: AdditiveNoncollidedTransportResponse,
    layout: MeanCalibrationLayout,
    design: Mapping[str, object],
) -> Path:
    """Freeze a runtime model after authenticating every active component.

    Fixed-quota mean calibration cannot identify station-level count
    dispersion.  A model may either use the exact physical count/mark law with
    no empirical discrepancy, or supply an independently authenticated
    discrepancy contract.  Its additive response must be the response fitted
    from this exact predeclared calibration corpus.
    """
    resolved = validate_predeclared_mean_calibration_design(design)
    try:
        expected_completion = build_mean_calibration_completion_manifest(
            layout=layout,
            design=resolved,
        )
        observed_completion = json.loads(
            layout.completion_path.read_text(encoding="utf-8")
        )
        observed_additive = json.loads(
            layout.additive_model_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Runtime freeze requires the complete predeclared mean-calibration "
            "training corpus and its frozen additive response."
        ) from exc
    if (
        observed_completion != expected_completion
        or observed_additive != additive_response.to_payload()
    ):
        raise RuntimeError(
            "Runtime freeze rejected stale mean-calibration training "
            "provenance."
        )
    active = model.additive_scatter_response
    if (
        active is None
        or active.contract_hash_sha256
        != additive_response.contract_hash_sha256
        or not model.runtime_ready
    ):
        raise RuntimeError(
            "Runtime freeze requires this calibrated additive response plus "
            "either exact physical statistics or an independently "
            "authenticated station-discrepancy contract."
        )
    return _write_immutable_bytes(
        Path(output_path),
        canonical_json_bytes(model.manifest_payload()),
    )


__all__ = [
    "ExternalGeant4MeanCalibrationBackend",
    "MEAN_CALIBRATION_DESIGN_SCHEMA_VERSION",
    "MEAN_CALIBRATION_MODEL_ID",
    "MeanCalibrationLayout",
    "MeanCalibrationTrainingRow",
    "build_mean_calibration_app_payload",
    "build_mean_calibration_completion_manifest",
    "build_mean_calibration_scene_manifest",
    "build_predeclared_mean_calibration_design",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "file_sha256",
    "fit_additive_scatter_from_complete_mean_calibration",
    "fit_additive_scatter_training_rows",
    "freeze_mean_calibration_completion_manifest",
    "freeze_mean_calibration_scene_manifest",
    "freeze_runtime_ready_model",
    "initialize_mean_calibration_layout",
    "load_mean_calibration_pair_artifact",
    "validate_predeclared_mean_calibration_design",
    "write_mean_calibration_pair_artifact",
]
