"""Export actual Geant4 gamma trajectories for one authenticated RA-L scene.

The diagnostic runs the native sidecar twice at one recorded detector pose:
an analog isotropic-emission sample supplies sparse outward tracks from every
source, and a fixed-quota detector-cone sample supplies detector-entering
tracks. Both outputs contain Geant4 step endpoints, not illustrative rays.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_PF_ROOT = ROOT.parent / "Rotating-shield-particle-filter"
DEFAULT_DIAGNOSTIC_EXECUTABLE = ROOT / "build" / "geant4_trajectory_sidecar"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim.geant4_app.app import Geant4Application  # noqa: E402
from sim.isaacsim_app.scene_builder import build_scene_description  # noqa: E402
from sim.protocol import SimulationCommand  # noqa: E402
from runtime.measurement_log import load_measurement_log  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    """Return a stable path, size, and hash record."""
    resolved = Path(path).expanduser().resolve()
    return {
        "path": resolved.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    """Return planar yaw from one WXYZ quaternion."""
    w_value, x_value, y_value, z_value = (
        float(value) for value in np.asarray(quaternion, dtype=np.float64)
    )
    return math.atan2(
        2.0 * (w_value * z_value + x_value * y_value),
        1.0 - 2.0 * (y_value * y_value + z_value * z_value),
    )


def _parse_fields(tokens: list[str]) -> dict[str, str]:
    """Parse strict key-value tokens from the native text format."""
    fields: dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator or not key or not value or key in fields:
            raise ValueError(f"Malformed Geant4 trajectory token: {token!r}.")
        fields[key] = value
    return fields


def _parse_bool(value: str) -> bool:
    """Parse one native zero-or-one boolean."""
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid native boolean value {value!r}.")
    return value == "1"


def parse_trajectory_file(path: Path, *, mode: str) -> dict[str, object]:
    """Parse a native Geant4 step-trajectory file without altering points."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    expected_format = "FORMAT geant4_primary_gamma_step_trajectory_v1"
    if not lines or lines[0] != expected_format:
        raise ValueError(f"{path} has an unsupported trajectory format.")
    metadata: dict[str, str] = {}
    tracks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line_number, line in enumerate(lines[1:], start=2):
        tokens = line.split()
        if not tokens:
            continue
        record_type = tokens[0]
        fields = _parse_fields(tokens[1:])
        if record_type == "META":
            if (
                len(fields) != 1
                or current is not None
                or metadata.keys() & fields.keys()
            ):
                raise ValueError(f"{path}:{line_number} has malformed metadata.")
            metadata.update(fields)
            continue
        if record_type == "TRACK":
            if current is not None:
                raise ValueError(f"{path}:{line_number} starts a nested track.")
            current = {
                "mode": mode,
                "source_index": int(fields["source_index"]),
                "isotope": fields["isotope"],
                "primary_batch_index": int(fields["primary_batch_index"]),
                "primary_history_index": int(fields["primary_history_index"]),
                "bias_branch_lineage_id": int(fields["bias_branch_lineage_id"]),
                "track_id": int(fields["track_id"]),
                "parent_id": int(fields["parent_id"]),
                "initial_energy_keV": float(fields["initial_energy_keV"]),
                "raw_step_count": int(fields["raw_step_count"]),
                "detector_entered": _parse_bool(fields["detector_entered"]),
                "interacted": _parse_bool(fields["interacted"]),
                "points_truncated": _parse_bool(fields["points_truncated"]),
                "declared_point_count": int(fields["point_count"]),
                "points_xyz_m": [],
            }
            continue
        if record_type == "POINT":
            if current is None:
                raise ValueError(f"{path}:{line_number} has an orphan point.")
            points = current["points_xyz_m"]
            if not isinstance(points, list) or int(fields["index"]) != len(points):
                raise ValueError(
                    f"{path}:{line_number} has a noncontiguous point index."
                )
            point = [float(fields[key]) for key in ("x_m", "y_m", "z_m")]
            if not np.all(np.isfinite(point)):
                raise ValueError(f"{path}:{line_number} contains a nonfinite point.")
            points.append(point)
            continue
        if record_type == "END_TRACK":
            if fields or current is None:
                raise ValueError(f"{path}:{line_number} has malformed track closure.")
            points = current["points_xyz_m"]
            if (
                not isinstance(points, list)
                or len(points) != current["declared_point_count"]
                or len(points) < 2
            ):
                raise ValueError(f"{path}:{line_number} closes an invalid track.")
            current.pop("declared_point_count")
            tracks.append(current)
            current = None
            continue
        raise ValueError(f"{path}:{line_number} has unknown record {record_type!r}.")
    if current is not None:
        raise ValueError(f"{path} ends inside a track record.")
    if int(metadata.get("recorded_track_count", "-1")) != len(tracks):
        raise ValueError(f"{path} track count disagrees with its metadata.")
    return {"metadata": metadata, "tracks": tracks}


def _recorded_station(
    observations_path: Path,
    station_index: int,
    *,
    run_id: str,
) -> tuple[np.ndarray, float, int]:
    """Select a station only from an authenticated log with the requested run ID."""
    if observations_path.name != "observations.npz":
        raise ValueError("Select observations.npz from a complete MeasurementLog.")
    log = load_measurement_log(observations_path.parent)
    if log.run_id != run_id:
        raise ValueError("MeasurementLog run ID disagrees with the requested scene.")
    record = next((row for row in log.records if row.station_id == station_index), None)
    if record is None:
        raise ValueError(f"Station {station_index} is absent from observations.")
    position = np.asarray(record.detector_pose_xyz, dtype=np.float64)
    quaternion = np.asarray(record.detector_quat_wxyz, dtype=np.float64)
    pair_id = record.fe_orientation_index * 8 + record.pb_orientation_index
    return position, _yaw_from_quaternion_wxyz(quaternion), pair_id


def _diagnostic_config(
    base: dict[str, Any],
    *,
    mode: str,
    trajectory_path: Path,
    executable_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Build a no-shortcut one-shot transport configuration for track export."""
    config = dict(base)
    config.update(
        {
            "use_mock_stage": True,
            "headless": True,
            "persistent_process": False,
            "thread_count": 1,
            "random_seed_base": int(seed),
            "executable_path": executable_path.resolve().as_posix(),
            "primary_emission_model": "independent_gamma_lines",
            "secondary_transport_mode": "full_transport",
            "primary_sampling_fraction": 1.0,
            "background_cps": 0.0,
            "dead_time_tau_s": 0.0,
            "sample_detector_response": False,
            "detector_green_operator_manifest": None,
            "executable_args": [
                "--geant4-trajectory-output",
                trajectory_path.resolve().as_posix(),
                "--geant4-trajectory-max-tracks-per-source",
                "256",
                "--geant4-trajectory-max-points-per-track",
                "512",
            ],
            "timeout_s": 600.0,
        }
    )
    for key in (
        "target_sampled_primaries",
        "mean_calibration_histories_per_source_line",
        "mean_calibration_angle_strata_mu",
        "mean_calibration_angle_strata_phi",
        "mean_calibration_forced_collision",
        "accelerated_weighted_transport_enable",
        "validation_entry_class_spectra",
    ):
        config.pop(key, None)
    if mode == "isotropic_emission":
        config.update(
            {
                "source_rate_model": "isotropic_emission_equivalent",
                "source_bias_mode": "analog",
                "source_bias_isotropic_fraction": 1.0,
                "detector_scoring_mode": "full_transport",
                "validation_entry_class_spectra": False,
            }
        )
    elif mode == "detector_entering":
        config.update(
            {
                "source_rate_model": "detector_cps_1m",
                "source_bias_mode": "detector_cone",
                "source_bias_isotropic_fraction": 1.0,
                "detector_scoring_mode": "incident_gamma_energy",
                "mean_calibration_histories_per_source_line": 128,
                "mean_calibration_angle_strata_mu": 4,
                "mean_calibration_angle_strata_phi": 4,
                "mean_calibration_forced_collision": False,
                "validation_entry_class_spectra": True,
            }
        )
    else:
        raise ValueError(f"Unsupported trajectory diagnostic mode {mode!r}.")
    return config


def _run_mode(
    *,
    mode: str,
    base_config: dict[str, Any],
    scene_payload: dict[str, Any],
    command: SimulationCommand,
    trajectory_path: Path,
    executable_path: Path,
    seed: int,
) -> dict[str, object]:
    """Run one actual Geant4 diagnostic mode and parse its track output."""
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_path.exists():
        raise FileExistsError(
            f"Refusing to replace trajectory output: {trajectory_path}."
        )
    app = Geant4Application(
        app_config=_diagnostic_config(
            base_config,
            mode=mode,
            trajectory_path=trajectory_path,
            executable_path=executable_path,
            seed=seed,
        )
    )
    try:
        app.reset(build_scene_description(scene_payload))
        observation = app.step(command)
    finally:
        app.close()
    if not trajectory_path.is_file():
        raise RuntimeError(f"Geant4 did not write {trajectory_path}.")
    parsed = parse_trajectory_file(trajectory_path, mode=mode)
    metadata = dict(observation.metadata)
    return {
        **parsed,
        "observation": {
            "total_spectrum_counts": float(np.sum(observation.spectrum_counts)),
            "metadata": metadata,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse authenticated run inputs and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--station-index", type=int, required=True)
    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_EXECUTABLE,
        help=(
            "Dedicated trajectory-diagnostic sidecar built with "
            "scripts/build_geant4_sidecar.py --trajectory-diagnostic."
        ),
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Private runtime configuration; defaults to the selected run's file.",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
    )
    parser.add_argument(
        "--truth-manifest",
        type=Path,
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="observations.npz inside the selected run's complete MeasurementLog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON output; defaults to a fresh private trajectory directory.",
    )
    args = parser.parse_args(argv)
    if (
        not args.run_id
        or args.run_id in {".", ".."}
        or Path(args.run_id).name != args.run_id
    ):
        parser.error("--run-id must be one nonempty path component.")
    if args.station_index < 0:
        parser.error("--station-index must be nonnegative.")
    private_root = ROOT / "private_runs" / "ral_ablation"
    for field, directory in (
        ("runtime_config", "runtime_configs"),
        ("scenario", "scenarios"),
        ("truth_manifest", "truth_manifests"),
    ):
        if getattr(args, field) is None:
            setattr(args, field, private_root / directory / f"{args.run_id}.json")
    if args.observations is None:
        args.observations = (
            DEFAULT_PF_ROOT
            / "results"
            / "ral_ablation"
            / "measurement_logs"
            / args.run_id
            / "observations.npz"
        )
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{args.run_id}_station_{args.station_index:02d}_{timestamp}_{uuid4().hex[:8]}"
        args.output = private_root / "figure_tracks" / name / "trajectories.json"
    return args


def _prepare_output_directory(output_path: Path) -> Path:
    """Reserve fresh raw output space without replacing completed or failed work."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace trajectory output: {output_path}.")
    raw_dir = output_path.parent / "raw" / output_path.stem
    raw_dir.mkdir(parents=True, exist_ok=False)
    return raw_dir


def main() -> None:
    """Run both trajectory diagnostics and write one provenance-rich artifact."""
    args = parse_args()
    runtime_config_path = Path(args.runtime_config).expanduser().resolve()
    scenario_path = Path(args.scenario).expanduser().resolve()
    truth_path = Path(args.truth_manifest).expanduser().resolve()
    observations_path = Path(args.observations).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    executable_path = Path(args.executable).expanduser().resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(
            f"Trajectory-diagnostic Geant4 executable not found: {executable_path}."
        )
    production_executable = (ROOT / "build" / "geant4_sidecar").resolve()
    if executable_path == production_executable:
        raise ValueError(
            "Trajectory export must use a dedicated diagnostic executable, not "
            "the independently approved production sidecar."
        )
    if not production_executable.is_file():
        raise FileNotFoundError(
            f"Production Geant4 executable not found: {production_executable}."
        )
    runtime_config = _read_json(runtime_config_path)
    scenario = _read_json(scenario_path)
    truth = _read_json(truth_path)
    if scenario.get("run_id") != args.run_id or truth.get("run_id") != args.run_id:
        raise ValueError("Run ID disagrees across the authenticated inputs.")
    scene_payload = scenario.get("scene")
    if not isinstance(scene_payload, dict):
        raise TypeError("scenario.scene must be an object.")
    position, yaw_rad, pair_id = _recorded_station(
        observations_path,
        int(args.station_index),
        run_id=args.run_id,
    )
    command = SimulationCommand(
        step_id=0,
        target_pose_xyz=tuple(float(value) for value in position),
        target_base_yaw_rad=float(yaw_rad),
        fe_orientation_index=pair_id // 8,
        pb_orientation_index=pair_id % 8,
        dwell_time_s=1.0e-7,
    )
    source_files = [
        runtime_config_path,
        scenario_path,
        truth_path,
        observations_path,
        ROOT / "native" / "geant4_sidecar" / "geant4_sidecar.cpp",
        ROOT / "scripts" / "build_geant4_sidecar.py",
        executable_path,
        Path(__file__),
    ]
    diagnostic_build_metadata = executable_path.with_name(
        f"{executable_path.name}.build.json"
    )
    if diagnostic_build_metadata.is_file():
        source_files.append(diagnostic_build_metadata)
    records = {
        path.resolve(): _artifact_record(path)
        for path in {*source_files, production_executable}
    }
    production_record = records[production_executable]
    diagnostic_record = records[executable_path]
    if production_record["sha256"] == diagnostic_record["sha256"]:
        raise RuntimeError(
            "Trajectory diagnostic and production executables unexpectedly match."
        )
    raw_dir = _prepare_output_directory(output_path)
    seeds = {
        "isotropic_emission": 748921,
        "detector_entering": 748963,
    }
    modes: dict[str, dict[str, object]] = {}
    for mode in ("isotropic_emission", "detector_entering"):
        mode_command = command
        if mode == "detector_entering":
            mode_command = SimulationCommand(
                step_id=0,
                target_pose_xyz=command.target_pose_xyz,
                target_base_yaw_rad=command.target_base_yaw_rad,
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                dwell_time_s=20.0,
            )
        modes[mode] = _run_mode(
            mode=mode,
            base_config=runtime_config,
            scene_payload=scene_payload,
            command=mode_command,
            trajectory_path=raw_dir / f"{mode}.tracks",
            executable_path=executable_path,
            seed=seeds[mode],
        )
    source_count = len(truth.get("sources", []))
    isotropic_tracks = modes["isotropic_emission"]["tracks"]
    directed_tracks = modes["detector_entering"]["tracks"]
    if not isinstance(isotropic_tracks, list) or not isinstance(directed_tracks, list):
        raise TypeError("Parsed track collections are malformed.")
    counts_by_source = {
        source_index: sum(
            int(track["source_index"] == source_index) for track in isotropic_tracks
        )
        for source_index in range(source_count)
    }
    if any(count < 2 for count in counts_by_source.values()):
        raise RuntimeError(
            "Analog Geant4 diagnostic did not retain two tracks from every source: "
            f"{counts_by_source}."
        )
    detector_entering_count = sum(
        int(bool(track["detector_entered"])) for track in directed_tracks
    )
    if detector_entering_count == 0:
        raise RuntimeError("Directed Geant4 diagnostic recorded no detector entry.")
    payload = {
        "schema_version": 1,
        "artifact_semantics": (
            "actual native Geant4 primary-gamma step endpoints; no drawn or "
            "interpolated particle histories"
        ),
        "acquisition_relationship": "diagnostic rerun at a recorded pose, not original acquisition tracks",
        "run_id": args.run_id,
        "station_index": int(args.station_index),
        "detector_pose_xyz_m": position.tolist(),
        "detector_yaw_rad": float(yaw_rad),
        "recorded_pair_id": int(pair_id),
        "orientation_pair": {"fe": pair_id // 8, "pb": pair_id % 8},
        "source_count": source_count,
        "seeds": seeds,
        "execution_boundary": {
            "production_executable": production_record,
            "trajectory_diagnostic_executable": diagnostic_record,
            "diagnostic_is_separate_from_production": True,
        },
        "source_files": [records[path.resolve()] for path in source_files],
        "raw_track_files": {
            mode: _artifact_record(raw_dir / f"{mode}.tracks") for mode in modes
        },
        "modes": modes,
        "validation": {
            "isotropic_track_count_by_source": counts_by_source,
            "detector_entering_track_count": detector_entering_count,
            "all_points_are_native_step_endpoints": True,
        },
    }
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {output_path}")
    print(
        "Actual Geant4 tracks: "
        f"isotropic={len(isotropic_tracks)}, "
        f"detector_entering={detector_entering_count}/"
        f"{len(directed_tracks)}"
    )


if __name__ == "__main__":
    main()
