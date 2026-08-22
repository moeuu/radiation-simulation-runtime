"""Run the predeclared Geant4 RDM-versus-line-basis distance diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectrum.decay_cascade_comparison import (
    DecayCascadeComparisonDesign,
    detector_model_from_runtime_config,
    load_decay_cascade_comparison_design,
    run_decay_cascade_comparison,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DESIGN = (
    _ROOT / "configs" / "validation" / "decay_cascade_comparison.json"
)
_DEFAULT_RUNTIME_CONFIG = (
    _ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
_DEFAULT_SIDECAR = _ROOT / "build" / "geant4_sidecar"
_DEFAULT_OUTPUT_ROOT = _ROOT / "results" / "decay_cascade_comparison"


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the independent diagnostic."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact Geant4 radioactive-decay cascades with the "
            "authenticated detector-cps independent-line basis. The command "
            "does not alter or launch the standard PF simulation."
        )
    )
    parser.add_argument("--design", type=Path, default=_DEFAULT_DESIGN)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=_DEFAULT_RUNTIME_CONFIG,
        help="Supplies the exact detector assembly used by the next PF run.",
    )
    parser.add_argument("--sidecar", type=Path, default=_DEFAULT_SIDECAR)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--pictures-dir",
        type=Path,
        default=Path.home() / "Pictures",
        help="Copy the run-specific review figure here; use '-' to disable.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional unique artifact directory name.",
    )
    return parser


def _load_json_object(path: Path) -> Mapping[str, object]:
    """Load one JSON object from disk."""
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _require_current_sidecar(executable: Path) -> Path:
    """Reject a native executable older than its C++ source."""
    resolved = executable.expanduser().resolve()
    source = _ROOT / "native" / "geant4_sidecar" / "geant4_sidecar.cpp"
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Geant4 sidecar not found: {resolved}. Build it with "
            "`uv run python scripts/build_geant4_sidecar.py --profile native`."
        )
    if resolved.stat().st_mtime_ns < source.stat().st_mtime_ns:
        raise RuntimeError(
            "Geant4 sidecar is older than its source. Rebuild it with "
            "`uv run python scripts/build_geant4_sidecar.py --profile native`."
        )
    return resolved


def _record_lookup(
    manifest: Mapping[str, object],
) -> dict[tuple[str, float, str], Mapping[str, object]]:
    """Index acquired case records by isotope, distance, and emission model."""
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, Sequence):
        raise TypeError("Comparison manifest cases must be a sequence.")
    return {
        (
            str(record["isotope"]),
            float(record["distance_m"]),
            str(record["emission_model"]),
        ): record
        for record in raw_cases
        if isinstance(record, Mapping)
    }


def _write_summary_csv(
    manifest: Mapping[str, object],
    *,
    output_path: Path,
) -> None:
    """Write one concise case-level comparison table."""
    analysis = manifest.get("analysis")
    if not isinstance(analysis, Mapping):
        raise TypeError("Comparison manifest analysis must be an object.")
    cases = analysis.get("case_results")
    if not isinstance(cases, Sequence):
        raise TypeError("Comparison case results must be a sequence.")
    fieldnames = (
        "isotope",
        "distance_m",
        "status",
        "rdm_detected_pulses",
        "independent_detected_pulses",
        "common_band_tv",
        "common_band_tv_lower_95",
        "common_band_tv_upper_95",
        "coincidence_excess_fraction",
        "coincidence_excess_lower_95",
        "coincidence_excess_upper_95",
        "maximum_isolated_sum_candidate_upper_95",
        "reason",
    )
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in cases:
            if not isinstance(record, Mapping):
                continue
            writer.writerow({field: record.get(field) for field in fieldnames})


def _plot_comparison(
    manifest: Mapping[str, object],
    design: DecayCascadeComparisonDesign,
    *,
    output_path: Path,
) -> None:
    """Plot normalized wide-band spectra for every isotope and distance."""
    records = _record_lookup(manifest)
    figure, axes = plt.subplots(
        len(design.isotopes),
        len(design.distances_m),
        figsize=(4.4 * len(design.distances_m), 2.9 * len(design.isotopes)),
        squeeze=False,
        sharex=True,
    )
    for row, isotope in enumerate(design.isotopes):
        for column, distance in enumerate(design.distances_m):
            axis = axes[row, column]
            for model, label, color in (
                ("geant4_radioactive_decay", "Geant4 RDM", "#d95f02"),
                ("independent_gamma_lines", "detector-cps lines", "#1b9e77"),
            ):
                record = records[(isotope, distance, model)]
                spectrum = np.asarray(
                    np.load(str(record["spectrum_path"])),
                    dtype=np.float64,
                )
                total = max(float(np.sum(spectrum)), 1.0)
                energy = np.arange(spectrum.size, dtype=np.float64) * 2.0
                axis.step(
                    energy,
                    np.maximum(spectrum / total, 1.0e-10),
                    where="mid",
                    linewidth=0.8,
                    alpha=0.9,
                    label=label,
                    color=color,
                )
            axis.axvline(1700.0, color="#666666", linestyle="--", linewidth=0.8)
            axis.set_yscale("log")
            axis.set_ylim(1.0e-7, 1.0)
            axis.set_xlim(0.0, design.energy_max_keV)
            axis.grid(alpha=0.18)
            axis.set_title(f"{isotope}, {distance:g} m", fontsize=9)
            if row == len(design.isotopes) - 1:
                axis.set_xlabel("Detector pulse energy [keV]", fontsize=8)
            if column == 0:
                axis.set_ylabel("Normalized pulse probability", fontsize=8)
            axis.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    figure.suptitle(
        "Decay-cascade diagnostic (dashed line: standard PF band edge)",
        fontsize=11,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Acquire, analyze, and render the full predeclared comparison."""
    args = _parser().parse_args()
    design_path = args.design.expanduser().resolve()
    runtime_config_path = args.runtime_config.expanduser().resolve()
    design = load_decay_cascade_comparison_design(design_path)
    runtime_payload = _load_json_object(runtime_config_path)
    detector = detector_model_from_runtime_config(runtime_payload)
    executable = _require_current_sidecar(args.sidecar)
    run_id = args.run_id or datetime.now().strftime("cascade_compare_%Y%m%d_%H%M%S")
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run-id must be one nonempty path component.")
    output_directory = args.output_root.expanduser().resolve() / run_id
    manifest = run_decay_cascade_comparison(
        design=design,
        executable=executable,
        detector_model=detector,
        output_directory=output_directory,
    )
    manifest_path = output_directory / "decay_cascade_comparison_manifest.json"
    summary_path = output_directory / "decay_cascade_comparison_summary.csv"
    figure_path = output_directory / "decay_cascade_comparison.png"
    _write_summary_csv(manifest, output_path=summary_path)
    _plot_comparison(manifest, design, output_path=figure_path)
    pictures_dir = args.pictures_dir
    picture_copy: Path | None = None
    if str(pictures_dir) != "-":
        resolved_pictures = pictures_dir.expanduser().resolve()
        resolved_pictures.mkdir(parents=True, exist_ok=True)
        picture_copy = resolved_pictures / f"{run_id}_decay_cascade_comparison.png"
        shutil.copy2(figure_path, picture_copy)
    analysis = manifest["analysis"]
    print(f"status={analysis['overall_status']}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"figure={figure_path}")
    if picture_copy is not None:
        print(f"picture={picture_copy}")


if __name__ == "__main__":
    main()
