"""Cross-language tests for the calibration-only first-collision core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

from sim.geant4_app.first_collision_oracle import (
    build_piecewise_first_collision_law,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SegmentCase:
    """Describe one homogeneous segment passed to the native probe."""

    token: int
    length_m: float
    cross_sections: tuple[float, ...]
    collision_uniform: float
    process_uniform: float
    collision_lineage: int
    survivor_lineage: int


@pytest.fixture(scope="session")
def first_collision_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the dependency-free native probe once for this test session."""

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.fail("g++ is required to validate the native collision core.")
    build_directory = tmp_path_factory.mktemp("first_collision_native")
    executable = build_directory / "first_collision_probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            (
                REPOSITORY_ROOT
                / "native/first_collision/first_collision_core.cpp"
            ).as_posix(),
            (
                REPOSITORY_ROOT
                / "native/first_collision/first_collision_probe.cpp"
            ).as_posix(),
            "-o",
            executable.as_posix(),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def _encode_probe_input(
    segments: list[SegmentCase],
    *,
    calibration_only: bool = True,
    initial_weight: float = 1.0,
    primary_history: int = 7001,
    initial_lineage: int = 9001,
) -> str:
    """Serialize one segmented path in the native probe's text protocol."""

    lines = [
        " ".join(
            (
                "PATH",
                str(int(calibration_only)),
                repr(initial_weight),
                str(primary_history),
                str(initial_lineage),
                str(len(segments)),
            )
        )
    ]
    for segment in segments:
        channel_fields: list[str] = []
        for process_index, cross_section in enumerate(
            segment.cross_sections
        ):
            channel_fields.extend(
                (str(100 + process_index), repr(cross_section))
            )
        lines.append(
            " ".join(
                (
                    "SEGMENT",
                    str(segment.token),
                    repr(segment.length_m),
                    str(len(segment.cross_sections)),
                    *channel_fields,
                    repr(segment.collision_uniform),
                    repr(segment.process_uniform),
                    str(segment.collision_lineage),
                    str(segment.survivor_lineage),
                )
            )
        )
    return "\n".join(lines) + "\n"


def _parse_probe_records(output: str) -> list[dict[str, str]]:
    """Parse key-value records emitted by the native collision probe."""

    records: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        assert fields and fields[0] in {"RESULT", "BRANCH"}
        record = {"record": fields[0]}
        for field in fields[1:]:
            key, value = field.split("=", maxsplit=1)
            record[key] = value
        records.append(record)
    return records


def _run_probe(
    executable: Path,
    segments: list[SegmentCase],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run the native probe for one path without hiding failures."""

    return subprocess.run(
        [executable.as_posix()],
        input=_encode_probe_input(segments, **kwargs),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _select_process(
    cross_sections: tuple[float, ...],
    uniform: float,
) -> int:
    """Return the categorical process index selected by the native rule."""

    target = uniform * math.fsum(cross_sections)
    cumulative = 0.0
    last_positive: int | None = None
    for index, cross_section in enumerate(cross_sections):
        if cross_section <= 0.0:
            continue
        last_positive = index
        cumulative += cross_section
        if target < cumulative:
            return index
    assert last_positive is not None
    return last_positive


def _assert_native_matches_oracle(
    executable: Path,
    segments: list[SegmentCase],
    *,
    initial_weight: float,
    primary_history: int = 7001,
    initial_lineage: int = 9001,
) -> None:
    """Compare every native branch statistic with the NumPy oracle."""

    completed = _run_probe(
        executable,
        segments,
        initial_weight=initial_weight,
        primary_history=primary_history,
        initial_lineage=initial_lineage,
    )
    assert completed.returncode == 0, completed.stderr
    records = _parse_probe_records(completed.stdout)
    result = records[0]
    assert result["record"] == "RESULT"
    assert int(result["standard_runtime_enabled"]) == 0
    assert int(result["primary"]) == primary_history
    assert float(result["initial_weight"]) == initial_weight

    cross_sections = np.asarray(
        [segment.cross_sections for segment in segments],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [segment.length_m for segment in segments],
        dtype=np.float64,
    )
    law = build_piecewise_first_collision_law(
        lengths,
        cross_sections,
        initial_weight=initial_weight,
    )
    assert float(result["total_tau"]) == pytest.approx(
        float(np.sum(law.optical_depths)),
        rel=2.0e-15,
        abs=0.0,
    )
    assert float(result["no_collision_probability"]) == pytest.approx(
        law.no_collision_probability,
        rel=2.0e-15,
        abs=0.0,
    )
    assert float(result["collision_probability"]) == pytest.approx(
        -math.expm1(-float(np.sum(law.optical_depths))),
        rel=2.0e-15,
        abs=0.0,
    )
    assert float(result["leaf_weight_sum"]) == pytest.approx(
        initial_weight,
        rel=2.0e-15,
        abs=2.0e-15,
    )

    branches = records[1:]
    collision_by_segment = {
        int(branch["segment_index"]): branch
        for branch in branches
        if branch["kind"] == "collision"
    }
    survivor_by_segment = {
        int(branch["segment_index"]): branch
        for branch in branches
        if branch["kind"] != "collision"
    }
    positive_segments = {
        index
        for index, total in enumerate(law.total_cross_sections)
        if total > 0.0 and law.optical_depths[index] > 0.0
    }
    assert set(collision_by_segment) == positive_segments
    assert set(survivor_by_segment) == set(range(len(segments)))

    parent_lineage = initial_lineage
    for index, segment in enumerate(segments):
        local_survival = math.exp(-float(law.optical_depths[index]))
        local_collision = -math.expm1(-float(law.optical_depths[index]))
        survivor = survivor_by_segment[index]
        assert survivor["kind"] == (
            "uncollided" if index == len(segments) - 1 else "survivor"
        )
        assert int(survivor["primary"]) == primary_history
        assert int(survivor["parent_lineage"]) == parent_lineage
        assert int(survivor["child_lineage"]) == segment.survivor_lineage
        assert int(survivor["segment_token"]) == segment.token
        assert float(survivor["parent_weight"]) == pytest.approx(
            initial_weight * law.survival_probabilities_before[index],
            rel=3.0e-15,
            abs=0.0,
        )
        assert float(survivor["local_probability"]) == pytest.approx(
            local_survival,
            rel=2.0e-15,
            abs=0.0,
        )
        assert float(survivor["cumulative_probability"]) == pytest.approx(
            law.survival_probabilities_after[index],
            rel=3.0e-15,
            abs=0.0,
        )
        assert float(survivor["branch_weight"]) == pytest.approx(
            initial_weight * law.survival_probabilities_after[index],
            rel=3.0e-15,
            abs=0.0,
        )
        assert int(survivor["has_collision"]) == 0

        if index in positive_segments:
            collision = collision_by_segment[index]
            total_cross_section = float(
                law.total_cross_sections[index]
            )
            collision_tau = -math.log1p(
                -segment.collision_uniform * local_collision
            )
            distance_m = collision_tau / total_cross_section
            process_index = _select_process(
                segment.cross_sections,
                segment.process_uniform,
            )
            process_cross_section = segment.cross_sections[process_index]
            local_analog_density = (
                total_cross_section * math.exp(-collision_tau)
            )
            assert int(collision["primary"]) == primary_history
            assert int(collision["parent_lineage"]) == parent_lineage
            assert (
                int(collision["child_lineage"])
                == segment.collision_lineage
            )
            assert int(collision["segment_token"]) == segment.token
            assert float(collision["local_probability"]) == pytest.approx(
                local_collision,
                rel=2.0e-15,
                abs=0.0,
            )
            assert float(
                collision["cumulative_probability"]
            ) == pytest.approx(
                law.segment_collision_probabilities[index],
                rel=3.0e-15,
                abs=0.0,
            )
            assert float(collision["branch_weight"]) == pytest.approx(
                law.forced_segment_weights[index],
                rel=3.0e-15,
                abs=0.0,
            )
            assert float(collision["distance_m"]) == pytest.approx(
                distance_m,
                rel=3.0e-15,
                abs=0.0,
            )
            assert float(collision["collision_tau"]) == pytest.approx(
                collision_tau,
                rel=3.0e-15,
                abs=0.0,
            )
            assert int(collision["process_token"]) == 100 + process_index
            assert float(
                collision["process_probability"]
            ) == pytest.approx(
                process_cross_section / total_cross_section,
                rel=3.0e-15,
                abs=0.0,
            )
            assert float(
                collision["conditional_density_per_m"]
            ) == pytest.approx(
                local_analog_density / local_collision,
                rel=3.0e-15,
                abs=0.0,
            )
            assert float(
                collision["local_analog_density_per_m"]
            ) == pytest.approx(
                local_analog_density,
                rel=3.0e-15,
                abs=0.0,
            )
            assert float(
                collision["absolute_analog_density_per_m"]
            ) == pytest.approx(
                law.survival_probabilities_before[index]
                * local_analog_density,
                rel=3.0e-15,
                abs=0.0,
            )
        parent_lineage = segment.survivor_lineage


def test_native_core_matches_heterogeneous_python_oracle(
    first_collision_probe: Path,
) -> None:
    """Heterogeneous branch masses and samples must match the Python law."""

    segments = [
        SegmentCase(11, 2.0, (0.1, 0.2, 0.0), 0.125, 0.6, 101, 102),
        SegmentCase(12, 1.0, (0.0, 0.4, 0.1), 0.875, 0.95, 103, 104),
        SegmentCase(13, 0.75, (0.3, 0.0, 0.2), 0.5, 0.1, 105, 106),
    ]
    _assert_native_matches_oracle(
        first_collision_probe,
        segments,
        initial_weight=2.5,
        primary_history=8842,
        initial_lineage=100,
    )


def test_native_core_is_stable_for_vacuum_and_extreme_depths(
    first_collision_probe: Path,
) -> None:
    """Vacuum, tiny depth, and saturated depth must conserve exact mass."""

    segments = [
        SegmentCase(21, 3.0, (0.0, 0.0), 0.2, 0.8, 201, 202),
        SegmentCase(22, 1.0e-12, (1.0e-3, 0.0), 0.9, 0.2, 203, 204),
        SegmentCase(23, 1000.0, (0.2, 0.8), 0.999, 0.95, 205, 206),
    ]
    _assert_native_matches_oracle(
        first_collision_probe,
        segments,
        initial_weight=7.0,
        initial_lineage=200,
    )


def test_native_core_matches_randomized_heterogeneous_paths(
    first_collision_probe: Path,
) -> None:
    """Random deterministic paths must agree branch-for-branch with NumPy."""

    generator = random.Random(20260729)
    for path_index in range(20):
        segments: list[SegmentCase] = []
        lineage_base = 10_000 + path_index * 100
        for segment_index in range(generator.randint(1, 6)):
            cross_sections = tuple(
                0.0 if generator.random() < 0.2 else 10 ** generator.uniform(
                    -5.0,
                    1.0,
                )
                for _ in range(3)
            )
            segments.append(
                SegmentCase(
                    token=1000 + segment_index,
                    length_m=10 ** generator.uniform(-4.0, 1.0),
                    cross_sections=cross_sections,
                    collision_uniform=generator.random(),
                    process_uniform=generator.random(),
                    collision_lineage=lineage_base + 2 * segment_index + 1,
                    survivor_lineage=lineage_base + 2 * segment_index + 2,
                )
            )
        _assert_native_matches_oracle(
            first_collision_probe,
            segments,
            initial_weight=10 ** generator.uniform(-3.0, 3.0),
            primary_history=50_000 + path_index,
            initial_lineage=lineage_base,
        )


def test_native_core_fails_closed_outside_calibration(
    first_collision_probe: Path,
) -> None:
    """The independent core must reject any standard-runtime invocation."""

    segment = SegmentCase(31, 1.0, (0.2,), 0.5, 0.5, 301, 302)
    completed = _run_probe(
        first_collision_probe,
        [segment],
        calibration_only=False,
        initial_lineage=300,
    )

    assert completed.returncode == 2
    assert "calibration-only" in completed.stderr
    assert "not integrated with the standard Geant4 runtime" in (
        completed.stderr
    )


@pytest.mark.parametrize(
    ("segment", "initial_lineage", "message"),
    [
        (
            SegmentCase(41, 1.0, (-0.1,), 0.5, 0.5, 401, 402),
            400,
            "finite and nonnegative",
        ),
        (
            SegmentCase(42, 1.0, (0.1,), 1.0, 0.5, 403, 404),
            400,
            "collision_unit_interval",
        ),
        (
            SegmentCase(43, 1.0, (0.1,), 0.5, -0.1, 405, 406),
            400,
            "process_unit_interval",
        ),
        (
            SegmentCase(44, 1.0, (0.1,), 0.5, 0.5, 400, 408),
            400,
            "globally unique",
        ),
        (
            SegmentCase(45, 1.0, (0.1,), 0.5, 0.5, 409, 409),
            400,
            "globally unique",
        ),
    ],
)
def test_native_core_rejects_invalid_physical_or_lineage_inputs(
    first_collision_probe: Path,
    segment: SegmentCase,
    initial_lineage: int,
    message: str,
) -> None:
    """Invalid physics domains and lineage collisions must fail fast."""

    completed = _run_probe(
        first_collision_probe,
        [segment],
        initial_lineage=initial_lineage,
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_native_core_rejects_duplicate_process_tokens(
    first_collision_probe: Path,
) -> None:
    """Ambiguous process lineage tokens must be rejected explicitly."""

    payload = "\n".join(
        (
            "PATH 1 1.0 70 80 1",
            "SEGMENT 90 1.0 2 100 0.1 100 0.2 0.5 0.5 81 82",
            "",
        )
    )
    completed = subprocess.run(
        [first_collision_probe.as_posix()],
        input=payload,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "process tokens must be unique" in completed.stderr


def test_first_collision_core_is_not_wired_to_standard_runtime() -> None:
    """The unfinished Geant4 process wrapper must remain impossible to select."""

    header = (
        REPOSITORY_ROOT / "native/first_collision/first_collision_core.hh"
    ).read_text(encoding="utf-8")
    sidecar = (
        REPOSITORY_ROOT / "native/geant4_sidecar/geant4_sidecar.cpp"
    ).read_text(encoding="utf-8")
    build_script = (
        REPOSITORY_ROOT / "scripts/build_geant4_sidecar.py"
    ).read_text(encoding="utf-8")

    assert "kStandardRuntimeIntegrationEnabled = false" in header
    assert "first_collision_core" not in sidecar
    assert "first_collision_core" not in build_script
