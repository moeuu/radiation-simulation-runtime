"""Tests for shared truth-free CUI routing and browser serving."""

from __future__ import annotations

from dataclasses import replace
import errno
from http.client import HTTPConnection
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pytest
import runtime.cui as cui_module
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid

from runtime import (
    CUIAcquisitionFrame,
    CUIDashboardConfig,
    CUIPanelSpec,
    CUIRoute,
    CUIScene,
    CUIServerHandle,
    CUIStatus,
    CUITruthDisplayMode,
    CUI_URL_MESSAGE_PREFIX,
    cui_browser_url,
    cui_route_from_records,
    shared_cui_panel_specs,
    resolve_cui_public_host,
    start_cui_server,
    write_cui_index,
    write_cui_status,
)
from tests.runtime_test_support import records


def _fetch_text(url: str) -> str:
    """Fetch one local test page as UTF-8 text."""
    with urlopen(url, timeout=2.0) as response:
        return response.read().decode("utf-8")


def _wait_until_closed(host: str, port: int) -> bool:
    """Wait briefly for a terminated local HTTP endpoint to close."""
    for _ in range(40):
        try:
            with socket.create_connection((host, port), timeout=0.05):
                pass
        except OSError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"serve": 1}, TypeError),
        ({"host": ""}, TypeError),
        ({"host": "http://localhost"}, ValueError),
        ({"port": True}, TypeError),
        ({"port": -1}, ValueError),
        ({"port": 65536}, ValueError),
        ({"public_host": "localhost:9000"}, ValueError),
    ),
)
def test_dashboard_config_rejects_coerced_or_url_shaped_values(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    """CUI settings must preserve exact boolean, host, and port semantics."""
    with pytest.raises(error):
        CUIDashboardConfig(**kwargs)


def test_dashboard_config_parses_existing_cross_repository_keys() -> None:
    """The shared parser should retain the existing PF configuration names."""
    config = CUIDashboardConfig.from_mapping(
        {
            "cui_split_view_serve": False,
            "cui_split_view_host": "::1",
            "cui_split_view_port": 0,
            "cui_split_view_public_host": "2001:db8::8",
            "unrelated_runtime_setting": 10,
        }
    )

    assert config == CUIDashboardConfig(
        serve=False,
        host="::1",
        port=0,
        public_host="2001:db8::8",
    )


def test_browser_url_brackets_ipv6_and_quotes_nested_index() -> None:
    """Browser URLs must retain IPv6 authority and arbitrary index paths."""
    url = cui_browser_url(
        "2001:db8::1",
        8877,
        Path("run one") / "dashboard.html",
    )

    assert url == "http://[2001:db8::1]:8877/run%20one/dashboard.html"
    assert CUI_URL_MESSAGE_PREFIX == "CUI split visualization URL:"


@pytest.mark.parametrize(
    "index_path",
    (Path(".hidden/index.html"), Path("nested\\index.html")),
)
def test_browser_url_rejects_paths_the_server_will_not_expose(
    index_path: Path,
) -> None:
    """URL construction and safe static serving must share path semantics."""
    with pytest.raises(ValueError, match="visible relative path"):
        cui_browser_url("127.0.0.1", 8877, index_path)


def test_public_host_resolution_honors_explicit_env_and_bind_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit public host, concrete bind, and shared environment take priority."""
    monkeypatch.setenv("CUI_SPLIT_VIEW_PUBLIC_HOST", "cui.example.test")

    assert resolve_cui_public_host("0.0.0.0", "public.example.test") == (
        "public.example.test"
    )
    assert resolve_cui_public_host("127.0.0.1") == "127.0.0.1"
    assert resolve_cui_public_host("0.0.0.0") == "cui.example.test"


def test_wildcard_host_discovery_preserves_bind_address_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic public hosts must match IPv4 and IPv6 wildcard families."""
    class Completed:
        """Provide deterministic mixed-family hostname output."""

        stdout = "100.64.0.8 2001:db8::20\n"

    def completed_run(*args: object, **kwargs: object) -> Completed:
        """Return deterministic mixed-family hostname output."""
        del args, kwargs
        return Completed()

    monkeypatch.delenv("CUI_SPLIT_VIEW_PUBLIC_HOST", raising=False)
    monkeypatch.setattr(cui_module.subprocess, "run", completed_run)

    assert resolve_cui_public_host("0.0.0.0") == "100.64.0.8"
    assert resolve_cui_public_host("::") == "2001:db8::20"


@pytest.mark.parametrize(
    ("bind_host", "expected"),
    (("0.0.0.0", "127.0.0.1"), ("::", "::1")),
)
def test_wildcard_host_discovery_falls_back_with_matching_family(
    monkeypatch: pytest.MonkeyPatch,
    bind_host: str,
    expected: str,
) -> None:
    """Failed discovery must use the matching-family loopback address."""
    def fail(*args: object, **kwargs: object) -> None:
        """Raise a deterministic local discovery failure."""
        del args, kwargs
        raise OSError("unavailable")

    monkeypatch.delenv("CUI_SPLIT_VIEW_PUBLIC_HOST", raising=False)
    monkeypatch.setattr(cui_module.subprocess, "run", fail)
    monkeypatch.setattr(cui_module.socket, "getaddrinfo", fail)
    monkeypatch.setattr(cui_module.socket, "socket", fail)

    assert resolve_cui_public_host(bind_host) == expected


def test_ipv4_discovery_retains_usable_low_priority_lan_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 172.16/12-only host must remain reachable when route probing is offline."""
    class Completed:
        """Provide one deterministic low-priority private address."""

        stdout = "172.20.1.25\n"

    def completed_run(*args: object, **kwargs: object) -> Completed:
        """Return the deterministic hostname address."""
        del args, kwargs
        return Completed()

    def fail_probe(*args: object, **kwargs: object) -> None:
        """Make resolver and route probing unavailable after hostname lookup."""
        del args, kwargs
        raise OSError("offline")

    monkeypatch.delenv("CUI_SPLIT_VIEW_PUBLIC_HOST", raising=False)
    monkeypatch.setattr(cui_module.subprocess, "run", completed_run)
    monkeypatch.setattr(cui_module.socket, "getaddrinfo", fail_probe)
    monkeypatch.setattr(cui_module.socket, "socket", fail_probe)

    assert resolve_cui_public_host("0.0.0.0") == "172.20.1.25"


@pytest.mark.parametrize(
    ("candidate", "family"),
    (
        ("0.0.0.0", socket.AF_INET),
        ("127.0.0.1", socket.AF_INET),
        ("169.254.1.2", socket.AF_INET),
        ("224.0.0.1", socket.AF_INET),
        ("::", socket.AF_INET6),
        ("::1", socket.AF_INET6),
        ("fe80::1", socket.AF_INET6),
        ("ff02::1", socket.AF_INET6),
    ),
)
def test_automatic_public_host_rejects_non_browser_addresses(
    candidate: str,
    family: socket.AddressFamily,
) -> None:
    """Wildcard discovery must reject unusable or interface-local addresses."""
    assert not cui_module._usable_discovered_address(candidate, family=family)


def test_server_class_detects_aaaa_only_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An AAAA-only bind hostname must select an IPv6 HTTP server socket."""
    def ipv6_results(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
        """Return one deterministic IPv6 resolver result."""
        del args, kwargs
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]

    monkeypatch.setattr(cui_module.socket, "getaddrinfo", ipv6_results)

    assert cui_module._server_class("ipv6-only.example").address_family == (
        socket.AF_INET6
    )


def test_managed_server_supports_nested_index_and_closes(
    tmp_path: Path,
) -> None:
    """Port zero should serve the matching nested page and close with its handle."""
    root = tmp_path / "static"
    index = root / "run one" / "dashboard.html"
    index.parent.mkdir(parents=True)
    index.write_text("managed-dashboard", encoding="utf-8")
    handle = start_cui_server(
        root,
        index_path=Path("run one") / "dashboard.html",
        config=CUIDashboardConfig(
            host="127.0.0.1",
            port=0,
            public_host="127.0.0.1",
        ),
    )
    try:
        assert handle.managed
        assert not handle.persistent
        assert handle.port is not None
        assert handle.url is not None
        assert _fetch_text(handle.url) == "managed-dashboard"
    finally:
        handle.close()

    assert _wait_until_closed("127.0.0.1", int(handle.port))


def test_managed_server_supports_ipv6_loopback_when_available(
    tmp_path: Path,
) -> None:
    """An IPv6 literal should bind and serve through a bracket-safe URL."""
    probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        probe.bind(("::1", 0))
    except OSError as exc:
        pytest.skip(f"IPv6 loopback is unavailable: {exc}")
    finally:
        probe.close()
    root = tmp_path / "static"
    root.mkdir()
    (root / "index.html").write_text("ipv6-dashboard", encoding="utf-8")
    handle = start_cui_server(
        root,
        config=CUIDashboardConfig(
            host="::1",
            port=0,
            public_host="::1",
        ),
    )
    assert handle.port is not None
    assert handle.url == f"http://[::1]:{handle.port}/index.html"
    connection = HTTPConnection("::1", handle.port, timeout=2.0)
    try:
        connection.request("GET", "/index.html")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read().decode("utf-8") == "ipv6-dashboard"
    finally:
        connection.close()
        handle.close()


def test_server_allows_renderer_to_publish_index_after_start(
    tmp_path: Path,
) -> None:
    """An asynchronous renderer may create a root-contained index after startup."""
    root = tmp_path / "static"
    root.mkdir()
    handle = start_cui_server(
        root,
        index_path="pending/index.html",
        config=CUIDashboardConfig(
            host="127.0.0.1",
            port=0,
            public_host="127.0.0.1",
        ),
    )
    try:
        assert handle.url is not None
        with pytest.raises(URLError):
            _fetch_text(handle.url)
        pending = root / "pending" / "index.html"
        pending.parent.mkdir()
        pending.write_text("published-later", encoding="utf-8")
        assert _fetch_text(handle.url) == "published-later"
    finally:
        handle.close()


def test_server_rejects_index_outside_static_root(tmp_path: Path) -> None:
    """A URL must never claim an index that its selected root cannot serve."""
    root = tmp_path / "static"
    root.mkdir()
    outside = tmp_path / "outside.html"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="inside root"):
        start_cui_server(
            root,
            index_path=outside,
            config=CUIDashboardConfig(host="127.0.0.1", port=0),
        )


def test_disabled_server_returns_a_nonserving_handle(tmp_path: Path) -> None:
    """Disabling serving should retain path identity without binding a port."""
    root = tmp_path / "static"
    root.mkdir()

    handle = start_cui_server(
        root,
        config=CUIDashboardConfig(
            serve=False,
            host="127.0.0.1",
            port=8877,
        ),
    )

    assert isinstance(handle, CUIServerHandle)
    assert handle.port is None
    assert handle.url is None
    assert not handle.managed
    assert not handle.persistent


def test_fixed_server_skips_unknown_occupied_port_and_closes(
    tmp_path: Path,
) -> None:
    """A fixed-port server must bind the next port and remain handle-managed."""
    root = tmp_path / "static"
    root.mkdir()
    (root / "index.html").write_text(
        "fixed-dashboard",
        encoding="utf-8",
    )
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    requested_port = int(occupied.getsockname()[1])
    handle: CUIServerHandle | None = None
    try:
        handle = start_cui_server(
            root,
            config=CUIDashboardConfig(
                host="127.0.0.1",
                port=requested_port,
                public_host="127.0.0.1",
            ),
        )
        assert not handle.persistent
        assert handle.managed
        assert handle.port is not None
        assert handle.port != requested_port
        assert handle.process_id is None
        assert _fetch_text(str(handle.url)) == "fixed-dashboard"

        handle.close()

        assert _wait_until_closed("127.0.0.1", handle.port)
    finally:
        occupied.close()
        if handle is not None:
            port = handle.port
            handle.terminate()
            if port is not None:
                assert _wait_until_closed("127.0.0.1", port)


def test_fixed_server_preserves_non_occupancy_bind_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource and configuration errors must not masquerade as port conflicts."""
    root = tmp_path / "static"
    root.mkdir()

    def fail_build(*args: object, **kwargs: object) -> None:
        """Raise one representative non-occupancy socket failure."""
        del args, kwargs
        raise OSError(errno.EMFILE, "too many open files")

    monkeypatch.setattr(cui_module, "_build_http_server", fail_build)

    with pytest.raises(OSError) as error:
        start_cui_server(
            root,
            config=CUIDashboardConfig(host="127.0.0.1", port=8877),
        )

    assert error.value.errno == errno.EMFILE


def test_server_closes_bound_socket_when_thread_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread startup failure must release the already-bound server socket."""
    root = tmp_path / "static"
    root.mkdir()
    server = cui_module._build_http_server(root, "127.0.0.1", 0)
    port = int(server.server_address[1])

    def fail_start(self: object) -> None:
        """Simulate a runtime that cannot create another thread."""
        del self
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(cui_module.threading.Thread, "start", fail_start)

    with pytest.raises(RuntimeError, match="thread unavailable"):
        cui_module._start_server_thread(
            server,
            root=root,
            index_path=root / "index.html",
            relative_index=Path("index.html"),
            host="127.0.0.1",
            display_host="127.0.0.1",
        )

    rebound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        rebound.bind(("127.0.0.1", port))
    finally:
        rebound.close()


def test_server_denies_directory_listing_dotfiles_and_symlinks(
    tmp_path: Path,
) -> None:
    """Static serving must expose only ordinary files contained by its root."""
    root = tmp_path / "static"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "index.html").write_text("safe", encoding="utf-8")
    (root / ".secret").write_text("hidden", encoding="utf-8")
    (nested / "asset.txt").write_text("asset", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    handle = start_cui_server(
        root,
        config=CUIDashboardConfig(
            host="127.0.0.1",
            port=0,
            public_host="127.0.0.1",
        ),
    )
    assert handle.url is not None
    try:
        assert _fetch_text(handle.url) == "safe"
        assert _fetch_text(urljoin(handle.url, "nested/asset.txt")) == "asset"
        for relative in ("./", "nested/", ".secret", "linked.txt"):
            with pytest.raises(HTTPError) as error:
                _fetch_text(urljoin(handle.url, relative))
            assert error.value.code == 404
    finally:
        handle.terminate()


def test_route_normalizes_waypoints_station_visits_and_latest_observation() -> None:
    """Shared CUI data should reproduce PF path and station display semantics."""
    base = records(3)
    first_segment = [[0.0, 0.0, 0.4], [0.1, 0.2, 0.4], [0.25, 0.25, 0.4]]
    second_segment = [[0.25, 0.25, 0.4], [0.75, 0.5, 0.4]]
    inputs = (
        replace(
            base[0],
            metadata={
                **dict(base[0].metadata),
                "travel_waypoints_xyz": first_segment,
            },
        ),
        replace(
            base[1],
            metadata={
                **dict(base[1].metadata),
                "travel_waypoints_xyz": first_segment,
            },
        ),
        replace(
            base[2],
            metadata={
                **dict(base[2].metadata),
                "travel_waypoints_xyz": second_segment,
            },
        ),
    )

    route = cui_route_from_records(inputs)

    assert isinstance(route, CUIRoute)
    assert len(route.travel_path_segments_xyz) == 2
    np.testing.assert_allclose(route.path_segments_xyz[0], first_segment)
    np.testing.assert_allclose(
        route.measurement_stations_xyz,
        np.asarray([base[0].detector_pose_xyz, base[2].detector_pose_xyz]),
    )
    np.testing.assert_array_equal(route.measurement_station_ids, [0, 1])
    np.testing.assert_array_equal(route.measurement_step_ids, [0, 2])
    np.testing.assert_array_equal(route.measurement_visit_counts, [2, 1])
    np.testing.assert_allclose(
        route.current_pose_xyz,
        base[2].detector_pose_xyz,
    )
    np.testing.assert_array_equal(
        route.latest_spectrum_counts,
        base[2].spectrum_counts,
    )
    assert route.latest_step_id == 2
    assert "truth" not in json.dumps(route.to_payload(), sort_keys=True).lower()
    with pytest.raises(ValueError):
        route.measurement_stations_xyz[0, 0] = 100.0


def test_route_ignores_singleton_waypoints_and_rejects_bad_shape() -> None:
    """PF-compatible singleton paths are skipped while corrupt matrices fail."""
    base = records(1)[0]
    singleton = replace(
        base,
        metadata={
            **dict(base.metadata),
            "travel_waypoints_xyz": [[0.0, 0.0, 0.4]],
        },
    )
    assert cui_route_from_records((singleton,)).travel_path_segments_xyz == ()
    malformed = replace(
        base,
        metadata={
            **dict(base.metadata),
            "travel_waypoints_xyz": [[0.0, 0.0]],
        },
    )

    with pytest.raises(ValueError, match="shape"):
        cui_route_from_records((malformed,))


def test_empty_route_contains_only_immutable_empty_arrays() -> None:
    """An empty MeasurementLog prefix should remain renderable and truth-free."""
    route = cui_route_from_records(())

    assert route.travel_path_segments_xyz == ()
    assert route.measurement_stations_xyz.shape == (0, 3)
    assert route.measurement_visit_counts.shape == (0,)
    assert route.current_detector_position_xyz is None
    assert route.latest_spectrum_counts is None
    assert route.energy_bin_edges_keV is None
    assert route.latest_step_id is None
    assert not route.measurement_stations_xyz.flags.writeable


def test_route_requires_exact_records_in_causal_station_order() -> None:
    """CUI normalization must not reinterpret foreign or reordered records."""
    base = records(2)

    with pytest.raises(TypeError, match="exact MeasurementLogRecord"):
        cui_route_from_records((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="causal record order"):
        cui_route_from_records((replace(base[0], step_id=1),))
    with pytest.raises(ValueError, match="nondecreasing"):
        cui_route_from_records(
            (
                replace(base[0], station_id=1),
                replace(base[1], station_id=0),
            )
        )


def test_route_preserves_station_transition_at_the_same_pose() -> None:
    """Distinct runtime station IDs must not collapse solely by pose proximity."""
    base = records(2)
    inputs = (
        base[0],
        replace(base[1], station_id=1),
    )

    route = cui_route_from_records(inputs)

    np.testing.assert_array_equal(route.measurement_station_ids, [0, 1])
    np.testing.assert_array_equal(route.measurement_step_ids, [0, 1])
    np.testing.assert_array_equal(route.measurement_visit_counts, [1, 1])


def test_route_rejects_pose_changes_inside_one_station() -> None:
    """One runtime station cannot silently contain multiple detector poses."""
    base = records(2)
    moved = replace(base[1], detector_pose_xyz=(1.0, 1.0, 0.4))

    with pytest.raises(ValueError, match="share one pose"):
        cui_route_from_records((base[0], moved))


def test_scene_preserves_asymmetric_obstacle_xy_order() -> None:
    """Scene footprints must retain runtime (x, y) cell coordinates exactly."""
    environment = EnvironmentConfig(size_x=6.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.5, 1.0),
        cell_size=0.5,
        grid_shape=(4, 3),
        blocked_cells=((2, 1),),
    )

    scene = CUIScene.from_environment(
        environment,
        grid,
        obstacle_height_m=1.5,
    )

    np.testing.assert_allclose(
        scene.obstacle_boxes_xyz[0],
        [1.5, 1.5, 0.0, 2.0, 2.0, 1.5],
    )
    np.testing.assert_allclose(
        scene.obstacle_footprints_xy[0],
        [[1.5, 1.5], [2.0, 1.5], [2.0, 2.0], [1.5, 2.0]],
    )


def test_shared_shell_keeps_owner_defined_result_panels(
    tmp_path: Path,
) -> None:
    """Estimator result identity and count must remain owner-defined."""
    mle_results = (
        CUIPanelSpec(
            "mle-grid",
            "Surface MLE grid",
            "latest_mle_grid.png",
        ),
        CUIPanelSpec(
            "mle-hotspots",
            "Surface MLE hotspots",
            "latest_mle_hotspots.png",
            2,
        ),
    )
    panels = shared_cui_panel_specs(mle_results)

    index = write_cui_index(tmp_path, panels, title="MLE dashboard")
    status = CUIStatus(
        phase="estimating",
        message="station complete",
        step_id=4,
        station_id=2,
    )
    status_path = write_cui_status(tmp_path / "status.json", status)

    markup = index.read_text(encoding="utf-8")
    offsets = [markup.index(f'id="{panel.panel_id}"') for panel in panels]
    assert offsets == sorted(offsets)
    assert tuple(panel.panel_id for panel in panels) == (
        "overview",
        "robot",
        "mle-grid",
        "mle-hotspots",
        "spectrum",
    )
    assert "latest_mle_grid.png" in markup
    assert "Particle filter" not in markup
    assert "height: calc(50vh - 70px)" in markup
    assert "object-fit: contain" in markup
    assert '<link rel="icon" href="data:,">' in markup
    assert json.loads(status_path.read_text(encoding="utf-8")) == (
        status.to_payload()
    )


def test_shared_shell_accepts_a_different_number_of_particle_result_panels(
    tmp_path: Path,
) -> None:
    """The shell must not force MLE grids into PF particle panel slots."""
    particle_results = (
        CUIPanelSpec("particles", "Particles", "latest_particles.png"),
        CUIPanelSpec(
            "particle-labels",
            "Particles with labels",
            "latest_particle_labels.png",
            2,
        ),
        CUIPanelSpec(
            "posterior-summary",
            "Posterior summary",
            "latest_posterior_summary.png",
            2,
        ),
    )
    panels = shared_cui_panel_specs(particle_results)

    index = write_cui_index(tmp_path, panels, title="PF dashboard")

    markup = index.read_text(encoding="utf-8")
    assert len(panels) == 6
    assert all(panel.image_filename in markup for panel in particle_results)


def test_shared_panel_structure_protects_context_panel_identifiers() -> None:
    """Estimator result slots cannot silently replace shared route context."""
    with pytest.raises(ValueError, match="must not replace shared context"):
        shared_cui_panel_specs(
            (CUIPanelSpec("robot", "Estimator robot", "result.png"),)
        )

    with pytest.raises(ValueError, match="requires a result panel"):
        shared_cui_panel_specs(())


def test_cui_shell_requires_nonempty_unique_panel_identifiers(
    tmp_path: Path,
) -> None:
    """Generic shell validation should not depend on estimator semantics."""
    panel = CUIPanelSpec("result", "Result", "result.png")
    with pytest.raises(ValueError, match="at least one"):
        write_cui_index(tmp_path, ())
    with pytest.raises(ValueError, match="identifiers must be unique"):
        write_cui_index(tmp_path, (panel, panel))


def test_shared_cui_shell_can_resolve_assets_from_a_parent_page(tmp_path: Path) -> None:
    """Estimator subpages must reuse the shell without copying root images."""
    page = write_cui_index(
        tmp_path / "pf",
        shared_cui_panel_specs(
            (CUIPanelSpec("particles", "Particles", "latest_particles.png"),)
        ),
        asset_base_href="../",
    )

    markup = page.read_text(encoding="utf-8")
    assert '<base href="../">' in markup
    assert 'src="latest_particles.png"' in markup
    with pytest.raises(ValueError, match="safe relative"):
        write_cui_index(
            tmp_path / "unsafe",
            shared_cui_panel_specs(
                (
                    CUIPanelSpec(
                        "particles",
                        "Particles",
                        "latest_particles.png",
                    ),
                )
            ),
            asset_base_href="https://example.invalid/",
        )


def test_acquisition_frame_declares_truth_mode_without_truth_payload() -> None:
    """Truth visibility is explicit while realized source values remain elsewhere."""
    frame = CUIAcquisitionFrame(
        route=cui_route_from_records(records(1)),
        status=CUIStatus(phase="running", message=""),
        truth_display_mode=CUITruthDisplayMode.HIDDEN,
    )

    payload = frame.to_payload()
    assert payload["truth_display_mode"] == "hidden"
    assert "source" not in json.dumps(payload, sort_keys=True).lower()


def test_shared_cui_truth_modes_match_estimator_cli_contract() -> None:
    """The shared enum must expose the three established estimator modes."""
    assert tuple(mode.value for mode in CUITruthDisplayMode) == (
        "hidden",
        "evaluation_live",
        "post_run",
    )


@pytest.mark.parametrize(
    "filename",
    ("../panel.png", ".hidden.png", "nested/panel.png", "panel.jpg"),
)
def test_panel_spec_rejects_nonportable_image_names(filename: str) -> None:
    """Panel files must remain directly serveable by the safe static server."""
    with pytest.raises(ValueError, match="visible PNG"):
        CUIPanelSpec("panel", "Panel", filename)
