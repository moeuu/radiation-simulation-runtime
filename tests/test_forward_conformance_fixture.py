"""Tests for the shared forward-response fixture parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.forward_conformance import (
    FORWARD_CONFORMANCE_CASE_ORDER,
    ForwardConformanceFixture,
    ForwardConformanceFixtureError,
)


def _fixture() -> dict[str, object]:
    """Return one compact valid fixture spanning every canonical axis."""
    return {
        "schema_version": 1,
        "units": {
            "distance": "m",
            "live_time": "s",
            "source_strength": "detector_cps_1m",
        },
        "isotopes": ["Cs-137"],
        "detector_poses": [
            {"pose_id": "pose", "xyz": [2.0, 0.0, 0.5], "live_time_s": 2.0}
        ],
        "source_points": [
            {"source_id": "source", "xyz": [0.0, 0.0, 0.0]}
        ],
        "obstacles": [
            {"obstacle_id": "empty", "boxes": []},
            {
                "obstacle_id": "concrete",
                "boxes": [
                    {
                        "min_xyz": [0.8, -0.2, 0.0],
                        "max_xyz": [1.2, 0.2, 1.2],
                        "material": "concrete",
                    }
                ],
            },
        ],
        "shield_program": {
            "pairing": "cartesian_product",
            "fe_orientation_indices": [0, 3],
            "pb_orientation_indices": [1, 6],
        },
        "required_case_order": list(FORWARD_CONFORMANCE_CASE_ORDER),
    }


def test_fixture_parser_builds_canonical_ids_and_obstacle_grid(tmp_path: Path) -> None:
    """Typed axes must preserve case order and runtime obstacle semantics."""
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_fixture()), encoding="utf-8")

    fixture = ForwardConformanceFixture.from_path(path)

    assert fixture.case_ids()[0] == (
        "Cs-137|pose=pose|fe=00|pb=01|source=source|obstacle=empty"
    )
    assert fixture.case_ids()[-1] == (
        "Cs-137|pose=pose|fe=03|pb=06|source=source|obstacle=concrete"
    )
    assert fixture.obstacle_grid("empty") is None
    grid = fixture.obstacle_grid("concrete")
    assert grid is not None
    assert grid.transport_boxes_m == ((0.8, -0.2, 0.0, 1.2, 0.2, 1.2),)
    assert grid.collision_boxes_m == grid.transport_boxes_m


def test_fixture_parser_rejects_zero_volume_boxes() -> None:
    """All providers must reject degenerate boxes with identical semantics."""
    payload = _fixture()
    obstacles = payload["obstacles"]
    assert isinstance(obstacles, list)
    box = obstacles[1]["boxes"][0]
    box["max_xyz"][2] = 0.0

    with pytest.raises(ForwardConformanceFixtureError, match="must exceed"):
        ForwardConformanceFixture.from_payload(payload)


def test_fixture_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """Shared fixture loading must use the runtime strict JSON policy."""
    path = tmp_path / "fixture.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        ForwardConformanceFixture.from_path(path)
