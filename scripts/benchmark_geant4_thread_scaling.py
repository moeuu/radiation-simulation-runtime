"""Benchmark Geant4 physical cores against SMT with identical transport."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Final

from measurement.shielding import (
    SHIELD_POSE_CONTRACT_ID,
    SHIELD_POSE_CONTRACT_SHA256,
)
from sim.geant4_app.io_format import NATIVE_ACTION_IDENTITY_CONTRACT_ID
from spectrum.detector_green_operator import DetectorGreenOperator


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_SCENE: Final[Path] = (
    ROOT / "tests" / "native" / "force_collision_thin_leaf.scene"
)
DEFAULT_OPERATOR_MANIFEST: Final[Path] = (
    ROOT / "src" / "spectrum" / "assets" / "detector_green_operator"
    / "manifest.json"
)
THREAD_COUNTS: Final[tuple[int, int]] = (16, 32)
PHYSICS_METADATA_KEYS: Final[tuple[str, ...]] = (
    "backend",
    "physics_profile",
    "reference_physics_list",
    "electromagnetic_physics_constructor",
    "production_cut_range_mm",
    "geant4_physics_contract_sha256",
    "detector_scoring_mode",
    "secondary_transport_mode",
    "source_rate_model",
    "source_bias_mode",
    "primary_sampling_fraction",
    "history_thinning_enabled",
    "transport_history_mode",
    "scene_hash",
    "num_primaries",
    "total_track_steps",
)


@dataclass(frozen=True)
class SidecarResponse:
    """Hold native metadata and spectrum statistics for one transport run."""

    metadata: dict[str, str]
    spectrum: tuple[float, ...]
    variance: tuple[float, ...]


@dataclass(frozen=True)
class BenchmarkRun:
    """Hold one measured same-physics Geant4 execution."""

    thread_count: int
    cpu_affinity: tuple[int, ...]
    repeat_index: int
    process_wall_s: float
    num_primaries: int
    primaries_per_s: float
    effective_entries_per_s: float
    total_track_steps: int
    total_spectrum_counts: float
    spectrum_sha256: str


def _cpu_topology() -> dict[int, list[int]]:
    """Return logical CPUs grouped by physical core from Linux topology."""
    completed = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("lscpu is required for the Geant4 thread benchmark.")
    grouped: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_text, core_text, socket_text = line.split(",")
        if int(socket_text) != 0:
            raise RuntimeError("The thread benchmark requires one CPU socket.")
        grouped.setdefault(int(core_text), []).append(int(cpu_text))
    if len(grouped) < THREAD_COUNTS[0] or any(
        len(cpus) < 2 for cpus in grouped.values()
    ):
        raise RuntimeError(
            "The 16-core/32-thread benchmark requires at least 16 SMT cores."
        )
    return grouped


def _affinity_by_thread_count(topology: dict[int, list[int]]) -> dict[int, tuple[int, ...]]:
    """Return explicit physical-core and all-SMT logical CPU affinities."""
    cores = sorted(topology)[: THREAD_COUNTS[0]]
    physical = tuple(sorted(topology[core][0] for core in cores))
    smt = tuple(sorted(cpu for core in cores for cpu in topology[core][:2]))
    return {THREAD_COUNTS[0]: physical, THREAD_COUNTS[1]: smt}


def _write_request(
    path: Path,
    *,
    seed: int,
    dwell_time_s: float,
) -> None:
    """Write one immutable benchmark request shared by both thread counts."""
    identity = hashlib.sha256(
        f"geant4-thread-benchmark|{seed}|{dwell_time_s:.17g}".encode("utf-8")
    ).hexdigest()
    path.write_text(
        "\n".join(
            (
                (
                    f"STEP step_id=0 dwell_time_s={dwell_time_s:.17g} "
                    f"seed={seed} "
                    f"shield_pose_contract_id={SHIELD_POSE_CONTRACT_ID} "
                    "shield_pose_contract_sha256="
                    f"{SHIELD_POSE_CONTRACT_SHA256} "
                    "native_action_contract_id="
                    f"{NATIVE_ACTION_IDENTITY_CONTRACT_ID} "
                    f"native_action_sha256={identity} "
                    "fe_orientation_index=7 pb_orientation_index=7"
                ),
                "POSE kind=detector x=3 y=0 z=0 qw=1 qx=0 qy=0 qz=0",
                "POSE kind=fe x=3 y=0 z=0 qw=1 qx=0 qy=0 qz=0",
                "POSE kind=pb x=3 y=0 z=0 qw=1 qx=0 qy=0 qz=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_response(path: Path) -> SidecarResponse:
    """Parse one native sidecar response without weakening validations."""
    metadata: dict[str, str] = {}
    spectrum: tuple[float, ...] = ()
    variance: tuple[float, ...] = ()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("META "):
            key, value = line[5:].split("=", maxsplit=1)
            metadata[key] = value
        elif line.startswith("SPECTRUM "):
            spectrum = tuple(float(value) for value in line[9:].split(","))
        elif line.startswith("SPECTRUM_VARIANCE "):
            variance = tuple(float(value) for value in line[18:].split(","))
    if not spectrum or len(variance) != len(spectrum):
        raise RuntimeError("Geant4 benchmark response is incomplete.")
    return SidecarResponse(metadata, spectrum, variance)


def _sidecar_command(
    *,
    executable: Path,
    scene: Path,
    request: Path,
    response: Path,
    operator_manifest: Path,
    thread_count: int,
    affinity: tuple[int, ...],
) -> list[str]:
    """Return one full-fidelity sidecar command with explicit CPU affinity."""
    operator = DetectorGreenOperator.from_artifact(operator_manifest)
    binary = operator_manifest.parent / "operator.bin"
    return [
        "taskset",
        "--cpu-list",
        ",".join(str(cpu) for cpu in affinity),
        executable.as_posix(),
        "--physics-profile",
        "balanced",
        "--threads",
        str(thread_count),
        "--dead-time-tau-s",
        "5.813e-9",
        "--source-rate-model",
        "detector_cps_1m",
        "--source-bias-mode",
        "detector_cone",
        "--detector-scoring-mode",
        "incident_gamma_energy",
        "--secondary-transport-mode",
        "full_transport",
        "--primary-sampling-fraction",
        "1",
        "--sample-detector-response",
        "--detector-green-operator-path",
        binary.as_posix(),
        "--detector-green-operator-binary-sha256",
        str(operator.binary_sha256),
        "--detector-green-operator-contract-sha256",
        str(operator.contract_hash_sha256),
        "--background-cps",
        "12",
        "--scene",
        scene.as_posix(),
        "--request",
        request.as_posix(),
        "--response",
        response.as_posix(),
    ]


def _spectrum_sha256(response: SidecarResponse) -> str:
    """Return a canonical digest of one spectrum and variance pair."""
    payload = json.dumps(
        {"spectrum": response.spectrum, "variance": response.variance},
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_same_physics(responses: list[SidecarResponse]) -> None:
    """Fail if any run changes physics or exceeds Monte Carlo agreement."""
    if not responses:
        raise RuntimeError("No Geant4 benchmark responses were produced.")
    reference = responses[0]
    for response in responses[1:]:
        for key in PHYSICS_METADATA_KEYS:
            if response.metadata.get(key) != reference.metadata.get(key):
                raise RuntimeError(
                    f"Geant4 benchmark changed same-physics field {key!r}."
                )
        difference = abs(
            math.fsum(response.spectrum) - math.fsum(reference.spectrum)
        )
        standard_error = math.sqrt(
            math.fsum(response.variance) + math.fsum(reference.variance)
        )
        if difference > 6.0 * max(1.0, standard_error):
            raise RuntimeError(
                "Thread-count spectra exceed the six-sigma Monte Carlo bound."
            )


def _summaries(runs: list[BenchmarkRun]) -> dict[str, object]:
    """Return robust throughput summaries and the measured winner."""
    result: dict[str, object] = {}
    medians: dict[int, float] = {}
    for thread_count in THREAD_COUNTS:
        selected = [run for run in runs if run.thread_count == thread_count]
        rates = [run.primaries_per_s for run in selected]
        walls = [run.process_wall_s for run in selected]
        if not rates:
            raise RuntimeError(f"Missing {thread_count}-thread measurements.")
        medians[thread_count] = statistics.median(rates)
        result[str(thread_count)] = {
            "median_primaries_per_s": medians[thread_count],
            "minimum_primaries_per_s": min(rates),
            "maximum_primaries_per_s": max(rates),
            "median_process_wall_s": statistics.median(walls),
        }
    speedup = medians[32] / medians[16]
    result["smt_over_physical_speedup"] = speedup
    result["recommended_thread_count"] = 32 if speedup > 1.0 else 16
    return result


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Publish one benchmark report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    """Run alternating 16-core and 32-SMT same-physics measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=Path,
        default=ROOT / "build" / "geant4_sidecar",
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--operator-manifest",
        type=Path,
        default=DEFAULT_OPERATOR_MANIFEST,
    )
    parser.add_argument("--dwell-time-s", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=642901)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.expanduser().resolve()
    scene = args.scene.expanduser().resolve()
    operator_manifest = args.operator_manifest.expanduser().resolve()
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not scene.is_file()
        or not operator_manifest.is_file()
    ):
        raise SystemExit("Benchmark executable, scene, or operator is unavailable.")
    if (
        not math.isfinite(args.dwell_time_s)
        or args.dwell_time_s <= 0.0
        or args.warmups < 0
        or args.repeats < 2
    ):
        raise SystemExit("Benchmark dwell/repeat settings are invalid.")

    topology = _cpu_topology()
    affinities = _affinity_by_thread_count(topology)
    runs: list[BenchmarkRun] = []
    measured_responses: list[SidecarResponse] = []
    with tempfile.TemporaryDirectory(prefix="geant4_thread_scaling_") as raw:
        temporary = Path(raw)
        request = temporary / "benchmark.request"
        _write_request(
            request,
            seed=args.seed,
            dwell_time_s=args.dwell_time_s,
        )
        schedule: list[tuple[int, bool, int]] = []
        for warmup_index in range(args.warmups):
            for thread_count in THREAD_COUNTS:
                schedule.append((thread_count, True, warmup_index))
        for repeat_index in range(args.repeats):
            order = THREAD_COUNTS if repeat_index % 2 == 0 else THREAD_COUNTS[::-1]
            for thread_count in order:
                schedule.append((thread_count, False, repeat_index))
        for run_index, (thread_count, warmup, repeat_index) in enumerate(schedule):
            response_path = temporary / f"response_{run_index}.txt"
            command = _sidecar_command(
                executable=executable,
                scene=scene,
                request=request,
                response=response_path,
                operator_manifest=operator_manifest,
                thread_count=thread_count,
                affinity=affinities[thread_count],
            )
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            process_wall_s = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    "Geant4 thread benchmark failed: "
                    f"{completed.stderr.strip()}"
                )
            response = _parse_response(response_path)
            if warmup:
                continue
            measured_responses.append(response)
            metadata = response.metadata
            runs.append(
                BenchmarkRun(
                    thread_count=thread_count,
                    cpu_affinity=affinities[thread_count],
                    repeat_index=repeat_index,
                    process_wall_s=process_wall_s,
                    num_primaries=int(metadata["num_primaries"]),
                    primaries_per_s=float(metadata["primaries_per_sec"]),
                    effective_entries_per_s=float(
                        metadata["effective_entries_per_sec"]
                    ),
                    total_track_steps=int(metadata["total_track_steps"]),
                    total_spectrum_counts=float(metadata["total_spectrum_counts"]),
                    spectrum_sha256=_spectrum_sha256(response),
                )
            )
    _validate_same_physics(measured_responses)
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "geant4_same_physics_physical_core_vs_smt",
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "operator_contract_sha256": str(
            DetectorGreenOperator.from_artifact(
                operator_manifest
            ).contract_hash_sha256
        ),
        "seed": args.seed,
        "dwell_time_s": args.dwell_time_s,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "physical_core_count": len(topology),
        "logical_cpu_count": sum(len(cpus) for cpus in topology.values()),
        "same_physics_metadata": {
            key: measured_responses[0].metadata.get(key)
            for key in PHYSICS_METADATA_KEYS
        },
        "runs": [asdict(run) for run in runs],
        "summary": _summaries(runs),
    }
    output = args.output.expanduser().resolve()
    _write_json_atomic(output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
