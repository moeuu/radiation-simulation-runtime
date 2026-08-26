"""Tests for raw-evidence detector-response production approval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectrum.detector_response_validation import (
    DETECTOR_RESPONSE_RAW_CORPUS_BASENAME,
    build_detector_response_validation_manifest,
    canonical_json_bytes,
    load_detector_response_validation_manifest,
    validate_detector_response_raw_corpus,
    validate_detector_response_validation_manifest,
)
from spectrum.detector_response_validation_runner import (
    run_detector_response_validation,
)
from tests.detector_response_test_support import (
    passing_detector_response_raw_corpus,
    write_passing_detector_response_validation,
)


def test_manifest_is_recomputed_from_canonical_raw_corpus(
    tmp_path: Path,
) -> None:
    """A passing manifest must load only with its exact raw evidence."""
    manifest_path = write_passing_detector_response_validation(
        tmp_path / "validation"
    )

    manifest = load_detector_response_validation_manifest(manifest_path)

    assert manifest["all_passed"] is True
    assert len(manifest["line_results"]) == 9


def test_metric_rewrite_is_rejected_even_when_it_still_passes(
    tmp_path: Path,
) -> None:
    """A plausible passing metric cannot replace the raw-derived value."""
    manifest_path = write_passing_detector_response_validation(
        tmp_path / "validation"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metrics"]["maximum_total_variation"]["value"] = 0.01
    manifest_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="do not match the raw corpus"):
        load_detector_response_validation_manifest(manifest_path)


def test_raw_corpus_rewrite_is_rejected_by_file_digest(tmp_path: Path) -> None:
    """Changing one raw response bin invalidates the derived manifest."""
    directory = tmp_path / "validation"
    manifest_path = write_passing_detector_response_validation(directory)
    raw_path = directory / DETECTOR_RESPONSE_RAW_CORPUS_BASENAME
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["line_corpora"][0]["observed_weighted_spectrum"][0] += 1.0
    payload["line_corpora"][0]["pulse_count_weighted"] += 1.0
    raw_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="raw corpus hash is stale"):
        load_detector_response_validation_manifest(manifest_path)


def test_nonfinite_raw_or_manifest_metric_fails_closed() -> None:
    """NaN cannot silently satisfy a less-than acceptance threshold."""
    corpus = passing_detector_response_raw_corpus()
    corpus["line_corpora"][0]["observed_weighted_spectrum"][0] = float("nan")
    with pytest.raises(ValueError, match="weighted spectrum is invalid"):
        validate_detector_response_raw_corpus(corpus)

    manifest = build_detector_response_validation_manifest(
        passing_detector_response_raw_corpus()
    )
    manifest["metrics"]["maximum_total_variation"]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON number"):
        validate_detector_response_validation_manifest(manifest)


def test_failed_physical_comparison_is_recorded_but_not_approved() -> None:
    """A completed failed evaluation remains inspectable and non-production."""
    corpus = passing_detector_response_raw_corpus()
    for entry in corpus["line_corpora"]:
        shifted = [0.0] * len(entry["observed_weighted_spectrum"])
        shifted[0] = float(corpus["histories_per_energy"])
        entry["observed_weighted_spectrum"] = shifted
    manifest = build_detector_response_validation_manifest(corpus)

    assert manifest["all_passed"] is False
    validate_detector_response_validation_manifest(
        manifest,
        require_passed=False,
    )
    with pytest.raises(ValueError, match="did not pass"):
        validate_detector_response_validation_manifest(manifest)


def test_formal_runner_requires_a_new_output_root(tmp_path: Path) -> None:
    """The formal runner never resumes or overwrites a stale artifact root."""
    output_root = tmp_path / "already-present"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="new empty output root"):
        run_detector_response_validation(
            runtime_config_path=tmp_path / "missing.json",
            output_root=output_root,
            repository_root=tmp_path,
        )
