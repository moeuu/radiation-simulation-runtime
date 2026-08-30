"""Scoped implementation provenance for detector Green artifacts.

The detector operator is independent of application scenes, validation seed
sets, isotope profiles, and PF logic.  Its implementation digest therefore
binds only code that can affect detector-corpus construction, validation, or
artifact interpretation.  Native physics and configuration are authenticated
separately by their own digests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final


DETECTOR_GREEN_IMPLEMENTATION_PATHS: Final = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("scripts/build_detector_green_operator.py"),
    Path("scripts/validate_detector_green_operator.py"),
    Path("src/measurement/model.py"),
    Path("src/measurement/source_boundary.py"),
    Path("src/runtime/experiment_profiles.py"),
    Path("src/runtime/randomness.py"),
    Path("src/sim/protocol.py"),
    Path("src/sim/runtime.py"),
    Path("src/sim/geant4_app/app.py"),
    Path("src/sim/geant4_app/engine.py"),
    Path("src/sim/geant4_app/execution_environment.py"),
    Path("src/sim/geant4_app/io_format.py"),
    Path("src/sim/geant4_app/scene_export.py"),
    Path("src/sim/isaacsim_app/scene_builder.py"),
    Path("src/spectrum/detector_green_construction.py"),
    Path("src/spectrum/detector_green_construction_runner.py"),
    Path("src/spectrum/detector_green_operator.py"),
    Path("src/spectrum/detector_green_provenance.py"),
    Path("src/spectrum/detector_green_validation.py"),
    Path("src/spectrum/detector_green_validation_runner.py"),
    Path("src/spectrum/geant4_physics.py"),
    Path("src/spectrum/library.py"),
    Path("src/spectrum/mean_calibration.py"),
    Path("src/spectrum/native_metadata.py"),
    Path("src/spectrum/response_matrix.py"),
    Path("src/spectrum/runtime_model_keys.py"),
)


def detector_green_implementation_bundle_sha256(
    repository_root: str | Path,
) -> str:
    """Hash the exact Python implementation inputs of detector Green work."""
    root = Path(repository_root).resolve()
    digest = hashlib.sha256()
    for relative_path in DETECTOR_GREEN_IMPLEMENTATION_PATHS:
        source = root / relative_path
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                "Detector Green implementation input is missing or is a "
                f"symlink: {source}."
            )
        encoded_path = relative_path.as_posix().encode("utf-8")
        raw = source.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


__all__ = [
    "DETECTOR_GREEN_IMPLEMENTATION_PATHS",
    "detector_green_implementation_bundle_sha256",
]
