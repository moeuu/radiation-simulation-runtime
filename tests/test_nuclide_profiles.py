"""Contracts for evaluated nuclides and material-aware isotope profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import generate_surface_sources
from measurement.surface_charts import build_surface_chart_geometry
from runtime_environment import build_runtime_obstacle_environment
from sim.runtime import load_runtime_config
from sim.geant4_app.app import Geant4AppConfig
from spectrum.isotope_profiles import (
    available_isotope_profiles,
    require_isotope_profile,
    resolve_isotope_selection,
)
from spectrum.library import default_library, nuclide_catalog_sha256
from spectrum.transport_spectral import (
    geometry_conditioned_model_from_runtime_config,
)


def test_evaluated_catalog_contains_requested_decommissioning_nuclides() -> None:
    """All requested alternatives must expose physical decay metadata."""
    library = default_library()
    requested = {"Eu-152", "Nb-94", "Cs-134", "Sb-125", "Am-241"}

    assert requested.issubset(library)
    assert library["Eu-152"].geant4_excitation_keV == pytest.approx(45.5998)
    assert library["Co-60"].mean_gamma_multiplicity > 1.99
    assert library["Nb-94"].mean_gamma_multiplicity > 1.99
    assert library["Eu-152"].half_life_s > 13.0 * 365.25 * 86_400.0
    assert len(nuclide_catalog_sha256()) == 64


def test_legacy_nuclides_expose_current_evaluated_decay_metadata() -> None:
    """Legacy truth nuclides must retain physical decay and placement data."""
    library = default_library()

    assert library["Cs-137"].half_life_s == pytest.approx(
        30.018 * 365.25 * 86_400.0
    )
    assert library["Co-60"].mean_gamma_multiplicity > 1.99
    eu154_energies = {
        round(line.energy_keV, 3) for line in library["Eu-154"].decay_lines
    }
    assert {123.071, 723.305, 1004.725, 1274.436}.issubset(
        eu154_energies
    )
    assert library["Eu-154"].eligible_materials == ("concrete",)
    assert (
        require_isotope_profile("fukushima_eu154").material_conditioning
        == "catalog_physical"
    )


def test_catalog_transport_lines_are_immutable() -> None:
    """A caller must not be able to mutate the authenticated global basis."""
    cobalt = default_library()["Co-60"]

    assert isinstance(cobalt.lines, tuple)
    with pytest.raises(AttributeError):
        cobalt.lines.append(cobalt.lines[0])


def test_legacy_transport_basis_is_separate_from_decay_probabilities() -> None:
    """The fixed RA-L response basis must not masquerade as decay branching."""
    cobalt = default_library()["Co-60"]

    assert [line.energy_keV for line in cobalt.lines] == [1173.0, 1332.0]
    assert [line.intensity for line in cobalt.lines] == [0.5, 0.5]
    assert [line.energy_keV for line in cobalt.decay_lines] == pytest.approx(
        [1173.228, 1332.492]
    )
    assert sum(line.intensity for line in cobalt.decay_lines) > 1.99


def test_named_profile_selects_one_isotope_set_without_ambiguity() -> None:
    """One profile name must resolve isotope and placement policy together."""
    names, profile = resolve_isotope_selection(
        profile_name="fukushima_eu152",
        explicit_isotopes=None,
        fallback_isotopes=("Cs-137",),
    )

    assert names == ("Cs-137", "Co-60", "Eu-152")
    assert profile == require_isotope_profile("fukushima_eu152")
    assert profile.material_conditioning == "catalog_physical"
    assert "fukushima_nb94" in available_isotope_profiles()
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_isotope_selection(
            profile_name="fukushima_eu152",
            explicit_isotopes=("Cs-137",),
            fallback_isotopes=("Cs-137",),
        )


@pytest.mark.parametrize("profile_name", available_isotope_profiles())
def test_every_isotope_profile_resolves_an_authenticated_pf_model(
    profile_name: str,
) -> None:
    """Each selectable truth profile must also define the exact PF line state."""
    root = Path(__file__).resolve().parents[1]
    standard_config = load_runtime_config(
        root
        / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    standard_config["isotope_experiment_profile"] = profile_name
    model = geometry_conditioned_model_from_runtime_config(
        standard_config,
        run_root=root,
    )
    model_isotopes = {
        str(line["isotope"])
        for line in model.line_identity
    }

    assert model_isotopes == set(require_isotope_profile(profile_name).isotopes)
    assert model.runtime_ready is True


def test_profile_registry_digest_fails_closed() -> None:
    """A stale or substituted profile registry must not select a PF model."""
    root = Path(__file__).resolve().parents[1]
    registry_path = (
        root
        / "configs/geant4/models/isotope_profile_model_registry_v1.json"
    )

    with pytest.raises(ValueError, match="registry SHA-256"):
        geometry_conditioned_model_from_runtime_config(
            {
                "isotope_experiment_profile": "fukushima_eu152",
                "full_spectrum_model_registry_path": str(registry_path),
                "full_spectrum_model_registry_file_sha256": "0" * 64,
            },
            run_root=root,
        )


def test_material_conditioned_sources_use_isotope_eligible_surface_area() -> None:
    """Nb-94 must sample metal while Eu-152 remains on concrete surfaces."""
    environment = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 2)),
        transport_boxes_m=(
            (1.0, 1.0, 0.0, 2.0, 2.0, 2.0),
            (2.0, 2.0, 0.0, 3.0, 3.0, 2.0),
        ),
    )
    profile = require_isotope_profile("fukushima_nb94")
    library = default_library()
    eligible = {
        isotope: library[isotope].eligible_materials
        for isotope in profile.isotopes
    }
    sources = generate_surface_sources(
        env=environment,
        obstacle_grid=grid,
        rng=np.random.default_rng(913),
        isotopes=("Eu-152", "Nb-94"),
        intensity_cps_1m=500_000.0,
        count=80,
        chart_max_edge_m=0.5,
        eligible_materials_by_isotope={
            "Eu-152": library["Eu-152"].eligible_materials,
            "Nb-94": eligible["Nb-94"],
        },
        transport_component_materials=("concrete", "steel"),
        room_surface_material="concrete",
    )
    atlas = build_surface_chart_geometry(
        environment,
        grid,
        max_edge_m=0.5,
    )

    niobium_faces = [
        atlas.face_ids[int(source.surface_chart_id)]
        for source in sources
        if source.isotope == "Nb-94"
    ]
    europium_faces = [
        atlas.face_ids[int(source.surface_chart_id)]
        for source in sources
        if source.isotope == "Eu-152"
    ]
    assert niobium_faces
    assert all(str(face).startswith("transport_component_1_") for face in niobium_faces)
    assert europium_faces
    assert all(
        not str(face).startswith("transport_component_1_")
        for face in europium_faces
    )


def test_material_conditioning_fails_without_compatible_surface() -> None:
    """An impossible Nb-94 material contract must stop before simulation."""
    environment = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)

    with pytest.raises(ValueError, match="No physically eligible"):
        generate_surface_sources(
            env=environment,
            obstacle_grid=None,
            rng=np.random.default_rng(7),
            isotopes=("Nb-94",),
            intensity_cps_1m=500_000.0,
            count=1,
            eligible_materials_by_isotope={
                "Nb-94": default_library()["Nb-94"].eligible_materials,
            },
            transport_component_materials=(),
            room_surface_material="concrete",
        )


def test_runtime_transport_tables_follow_selected_profile(tmp_path) -> None:
    """Runtime attenuation tables must contain exactly the selected nuclides."""
    profile = require_isotope_profile("fukushima_nb94")
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=41,
        attach_known_transport=True,
        obstacle_height_m=2.0,
        transport_isotopes=profile.isotopes,
    )

    assert environment.grid is not None
    expected = set(profile.isotopes)
    assert set(environment.grid.transport_mu_by_isotope) == expected
    assert set(environment.grid.transport_line_mu_by_isotope) == expected
    assert (
        set(environment.grid.transport_line_compton_mu_by_isotope)
        == expected
    )


def test_radioactive_decay_config_is_explicit_and_fail_closed() -> None:
    """Cascade transport must reject every incompatible shortcut contract."""
    payload = {
        "primary_emission_model": "geant4_radioactive_decay",
        "source_rate_model": "parent_decay_activity_bq",
        "source_bias_mode": "analog",
        "detector_scoring_mode": "full_transport",
        "secondary_transport_mode": "full_transport",
        "primary_sampling_fraction": 1.0,
        "sample_detector_response": False,
    }

    config = Geant4AppConfig.from_dict(payload)
    assert config.primary_emission_model == "geant4_radioactive_decay"
    with pytest.raises(ValueError, match="radioactive_decay requires"):
        Geant4AppConfig.from_dict(
            {
                **payload,
                "detector_scoring_mode": "incident_gamma_energy",
                "sample_detector_response": True,
            }
        )
