"""Source-level regression checks for Geant4 sidecar geometry conventions."""

import re
from pathlib import Path


def test_geant4_sidecar_uses_inverted_placement_rotation_for_placed_volumes() -> None:
    """Static volumes, shields, and detector housing should share placement rotation rules."""
    source = Path("native/geant4_sidecar/geant4_sidecar.cpp").read_text(encoding="utf-8")

    assert "QuaternionToShieldPlacementRotation" not in source
    assert "auto rotation = QuaternionToPlacementRotation(volume.qw, volume.qx, volume.qy, volume.qz);" in source
    assert "auto rotation = QuaternionToPlacementRotation(pose.qw, pose.qx, pose.qy, pose.qz);" in source
    assert "auto housing_rotation = QuaternionToPlacementRotation(" in source
    assert "rotation->invert();" in source
    assert "auto* existing_rotation = physical->GetRotation();" in source
    assert "*existing_rotation = *rotation;" in source
    assert "physical->SetRotation(rotation.release());" in source


def test_geant4_world_bounds_include_every_daughter_with_positive_margin() -> None:
    """The native world must enclose authored geometry rather than only room sizes."""
    source = Path("native/geant4_sidecar/geant4_sidecar.cpp").read_text(
        encoding="utf-8"
    )

    margin_match = re.search(
        r"constexpr double kWorldDaughterMarginM = ([0-9.]+);",
        source,
    )
    assert margin_match is not None
    assert float(margin_match.group(1)) > 0.0
    assert "world_half_extents_m_ = ComputeWorldHalfExtentsM();" in source
    assert "daughter_max_abs_m[axis] + kWorldDaughterMarginM" in source
    assert "for (const auto& volume : scene_->volumes)" in source
    assert "const G4ThreeVector rotated_corner" in source
    assert "for (const auto& triangle : volume.triangles)" in source
    assert "for (const auto& source : scene_->sources)" in source
    assert "request_->detector_pose.x" in source
    assert "scene_->fe_shield->outer_radius_m" in source
    assert "scene_->pb_shield->outer_radius_m" in source
    assert "ValidateMovablePoses(request)" in source
    assert "pose lies outside the persistent Geant4 world." in source


def test_geant4_mean_calibration_uses_fixed_stratified_source_line_quota() -> None:
    """Calibration must use unbiased fixed quotas without runtime shortcuts."""
    source = Path("native/geant4_sidecar/geant4_sidecar.cpp").read_text(
        encoding="utf-8"
    )

    for token in (
        "mean_calibration_histories_per_source_line",
        "mean_calibration_angle_strata_mu",
        "mean_calibration_angle_strata_phi",
        "RandomConeDirectionStratum",
        "fixed_source_line_stratified_mean_calibration",
        "expected_source_line_mean_divided_by_fixed_quota",
        "stratified_fixed_quota_sample_mean_covariance",
        "independent_mu_phi_stratum_sample_mean_cluster_",
        "sufficient_statistics_v1",
        "independent_mu_phi_stratum_original_history_",
        "branch_cluster_sufficient_statistics_v2",
        "mean_calibration_entry_histograms",
        "mean_calibration_entry_variance",
        "CalibrationFirstCollisionOperator",
        "G4BOptrForceCollision",
        "mean_calibration_force_collision_max_rel_weight_error",
    ):
        assert token in source
    assert (
        "Mean calibration requires at least two histories per "
        in source
    )
    assert (
        "Calibration force collision may attach only to "
        in source
    )
    assert "daughter-free homogeneous non-air material leaves." in source
    assert "if (mean_calibration_forced_collision_) {" in source
    assert "biasing_physics->Bias(\"gamma\")" in source
    assert (
        "Unsupported Geant4 sidecar option: "
        in source
    )
