"""Verify clean distributions with current runtime code and model assets."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("measurement", "runtime", "sim", "spectrum")


def _build(project: Path, output: Path, kind: str) -> Path:
    """Build a distribution without installing its runtime dependencies."""
    subprocess.run(
        ["uv", "build", f"--{kind}", "--out-dir", str(output)], cwd=project,
        check=True, capture_output=True, text=True, timeout=120.0,
    )
    files = tuple(output.glob("*.whl" if kind == "wheel" else "*.tar.gz"))
    assert len(files) == 1
    return files[0]


def _assert_contents(wheel: Path, sources: dict[str, bytes]) -> None:
    """Require current code and canonical assets with no build residue."""
    with zipfile.ZipFile(wheel) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    payload = {n: value for n, value in files.items() if ".dist-info/" not in n}
    assert payload == sources
    assert "runtime_environment.py" in payload
    assert "spectrum/assets/detector_green_operator/operator.bin" in payload
    entry_points = next(v for n, v in files.items() if n.endswith("/entry_points.txt"))
    assert b"rotating-shield-sim = runtime.cli:main" in entry_points


def test_distributions_ignore_old_builds_and_preserve_runtime_assets(
    tmp_path: Path,
) -> None:
    """Old builds must not affect direct or sdist-rebuilt wheels."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore"):
        shutil.copy2(ROOT / name, project / name)
    for name in ("src", "native", "configs", "source_layouts", "obstacle_layouts"):
        shutil.copytree(
            ROOT / name, project / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
    sources = {
        p.relative_to(project / "src").as_posix(): p.read_bytes()
        for package in PACKAGES
        for p in (project / "src" / package).rglob("*") if p.is_file()
    }
    sources["runtime_environment.py"] = (
        project / "src/runtime_environment.py"
    ).read_bytes()
    for relative in (
        "build/lib/runtime_defaults.py",
        "build/lib/spectrum/paired_all64_phase_space.py",
        "build/lib/runtime/retired_marker.py",
        "build/geant4_sidecar",
        "results/runs/old/observations.json",
        "private_runs/old/truth.json",
        "logs/old/console.log",
        "data/manchester_nuclear_assets/scene.usda",
        "tmp/preview.png",
        "src/runtime/__pycache__/retired_marker.cpython-312.pyc",
        "src/obsolete.egg-info/SOURCES.txt",
    ):
        stale = project / relative
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"must not be packaged\n")
    (project / "build/lib/runtime/__init__.py").write_bytes(
        b"raise RuntimeError('old')\n"
    )

    direct = _build(project, tmp_path / "direct", "wheel")
    _assert_contents(direct, sources)
    sdist = _build(project, tmp_path / "source-dist", "sdist")
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        forbidden = {
            "build", "results", "private_runs", "logs", "data", "tmp", "__pycache__"
        }
        assert all(not (set(Path(n).parts) & forbidden) for n in names)
        assert all(not any(p.endswith(".egg-info") for p in Path(n).parts)
                   for n in names)
        assert any(n.endswith("/native/geant4_sidecar/geant4_sidecar.cpp")
                   for n in names)
        assert any(n.endswith("/configs/geant4/models/isotope_profile_model_registry.json")
                   for n in names)
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    _assert_contents(_build(unpacked, tmp_path / "rebuilt", "wheel"), sources)

    installed = tmp_path / "installed"
    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--target", str(installed), str(direct)],
        check=True, capture_output=True, text=True, timeout=120.0,
    )
    subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
         "import runtime.cli, runtime_environment, spectrum; "
         "from spectrum.detector_green_operator import DetectorGreenOperator; "
         "assert Path(runtime.cli.__file__).is_relative_to(sys.argv[1]); "
         "assert Path(runtime_environment.__file__).is_relative_to(sys.argv[1]); "
         "asset = Path(spectrum.__file__).parent / 'assets/detector_green_operator/manifest.json'; "
         "DetectorGreenOperator.from_artifact(asset).require_runtime_ready()",
         str(installed)],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60.0,
    )
