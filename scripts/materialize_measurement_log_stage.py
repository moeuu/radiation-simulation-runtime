"""Materialize a completed MeasurementLog stream stage for diagnosis.

The source stage is copied through the runtime's verified resume reader.  The
source write-ahead log is never modified or removed, and no truth payload is
introduced into the canonical diagnostic artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from runtime.measurement_log import MeasurementLogStreamWriter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    """Parse the source stream stage and new canonical output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _run_id(stage: Path) -> str:
    """Read and validate the immutable run identifier from the first row."""
    metadata_path = stage / "observation_metadata.jsonl"
    first_line = metadata_path.open("r", encoding="utf-8").readline()
    payload = json.loads(first_line)
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Staged metadata has no valid run_id.")
    return run_id


def _current_commit() -> str:
    """Return the current runtime commit for resume provenance only."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    """Verify the stream stage and write an independent canonical bundle."""
    args = _parse_args()
    stage = args.stage.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}.")
    runtime_config = _load_json(stage / "runtime_config.resolved.json")
    environment = _load_json(stage / "environment.json")
    forward_model = _load_json(stage / "forward_model_manifest.input.json")
    repository_commit = (
        stage / "repository_commit.txt"
    ).read_text(encoding="utf-8").strip()
    isotopes = runtime_config.get("candidate_isotopes")
    if not isinstance(isotopes, list) or not all(
        isinstance(value, str) for value in isotopes
    ):
        raise ValueError("Staged runtime has no valid candidate_isotopes.")
    stage_name = stage.name
    suffix = stage_name.rfind(".stream-")
    if not stage_name.startswith(".") or suffix <= 1:
        raise ValueError("Stage directory does not follow the stream naming contract.")
    original_output = stage.parent / stage_name[1:suffix]
    resume_commit = _current_commit()
    writer = MeasurementLogStreamWriter.resume_from_stage(
        original_output,
        stage_dir=stage,
        run_id=_run_id(stage),
        repository_commit=repository_commit,
        runtime_config=runtime_config,
        environment=environment,
        forward_model_manifest=forward_model,
        isotopes=isotopes,
        metadata={"diagnostic_materialization": True},
        resume_execution_commit=resume_commit,
        resume_compatibility={
            "prefix_repository_commit": repository_commit,
            "resume_execution_commit": resume_commit,
            "purpose": "read_only_causal_diagnostic",
        },
    )
    materialized = writer.write_canonical_prefix(output)
    print(materialized.content_sha256)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
