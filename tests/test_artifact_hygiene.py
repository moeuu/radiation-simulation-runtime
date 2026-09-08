"""Check that ignored outputs stay organized and outside the Git index."""

from pathlib import Path
import subprocess

from scripts.audit_artifacts import ROOT, audit


def _repository(root: Path) -> None:
    """Create an isolated Git index with the repository's actual ignore rules."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())


def test_force_added_output_is_reported_without_deletion(tmp_path: Path) -> None:
    """Ignored data must be reported even after an accidental force-add."""
    _repository(tmp_path)
    output = tmp_path / "results" / "runs" / "run-id" / "observations.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    subprocess.run(["git", "add", "-f", str(output)], cwd=tmp_path, check=True)
    _, problems = audit(tmp_path)
    assert problems == [
        "Tracked generated/local file: results/runs/run-id/observations.json"
    ]
    assert output.read_text() == "{}"


def test_retired_trees_and_loose_outputs_are_reported(tmp_path: Path) -> None:
    """Gitignore must not hide old build copies or disorganized run outputs."""
    _repository(tmp_path)
    for name in ("build/lib", "results/old_trial", "results/previews", "logs"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "logs/loose.log").write_text("log")
    (tmp_path / "results/previews/loose.png").write_bytes(b"preview")
    _, problems = audit(tmp_path)
    assert "Retired directory: build/lib" in problems
    assert "Unexpected results category: results/old_trial" in problems
    assert "Loose output; use a run directory: logs/loose.log" in problems
    assert "Expected a run directory: results/previews/loose.png" in problems


def test_current_evidence_and_grouped_outputs_are_accepted(tmp_path: Path) -> None:
    """Keep approved evidence, native executables, and organized new runs."""
    _repository(tmp_path)
    for relative in (
        "results/full_spectrum_all64_acceptance/production_model.json",
        "results/detector_green_construction/run-id/logs/construction.log",
        "private_runs/ral_ablation/manifest.csv",
        "logs/session-id/bridge.log",
        "tmp/preview.png",
        "build/geant4_sidecar",
        "data/manchester_nuclear_assets/scene.usda",
    ):
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fixture")
    _, problems = audit(tmp_path)
    assert problems == []


def test_repository_index_excludes_generated_artifacts() -> None:
    """Check the real index in the suite without scanning local datasets."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-ci", "--exclude-standard"], cwd=ROOT, text=True,
    ).splitlines()
    assert tracked == [], f"Remove generated/local files from Git: {tracked}"


def test_default_previews_use_distinct_run_directories() -> None:
    """Repeated preview commands must not overwrite the preceding image."""
    from scripts.render_environment import build_parser

    first = build_parser().parse_args([]).output
    second = build_parser().parse_args([]).output
    assert first != second
    assert first.parent.parent == Path("results/previews")
    assert first.name == second.name == "environment.png"
    explicit = build_parser().parse_args(["--output", "chosen.png"]).output
    assert explicit == Path("chosen.png")
