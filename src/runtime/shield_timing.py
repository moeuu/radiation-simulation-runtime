"""Shared rotating-shield transition timing semantics."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import numpy as np

from measurement.shielding import generate_octant_orientations


DEFAULT_SHIELD_ANGULAR_SPEED_RAD_S = math.pi / 4.0
_SHIELD_PAIR_COUNT = 64


def _pair_id(value: object, *, name: str) -> int:
    """Return one exact canonical Fe/Pb pair identifier."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    parsed = int(value)
    if not 0 <= parsed < _SHIELD_PAIR_COUNT:
        raise ValueError(f"{name} must lie in [0, 63].")
    return parsed


def _angular_speed(value: object) -> float:
    """Return one finite positive shield angular speed."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("shield_angular_speed_rad_s must be a finite number.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(
            "shield_angular_speed_rad_s must be finite and positive."
        )
    return parsed


def shield_pair_transition_time_s(
    current_pair_id: int,
    target_pair_id: int,
    *,
    shield_angular_speed_rad_s: float,
) -> float:
    """Return parallel Fe/Pb actuation time for one pair transition."""
    current = _pair_id(current_pair_id, name="current_pair_id")
    target = _pair_id(target_pair_id, name="target_pair_id")
    speed = _angular_speed(shield_angular_speed_rad_s)
    orientations = np.asarray(generate_octant_orientations(), dtype=np.float64)
    current_fe, current_pb = divmod(current, 8)
    target_fe, target_pb = divmod(target, 8)

    def angle(first: int, second: int) -> float:
        """Return shortest angular displacement between octant normals."""
        cosine = float(
            np.clip(
                np.dot(orientations[first], orientations[second]),
                -1.0,
                1.0,
            )
        )
        return float(math.acos(cosine))

    displacement = max(
        angle(current_fe, target_fe),
        angle(current_pb, target_pb),
    )
    return displacement / speed


def shield_program_actuation_time_s(
    current_pair_id: int,
    pair_ids: Sequence[int],
    *,
    shield_angular_speed_rad_s: float,
) -> float:
    """Return total time for sequential transitions through a shield program."""
    current = _pair_id(current_pair_id, name="current_pair_id")
    if not isinstance(pair_ids, (list, tuple)) or not pair_ids:
        raise ValueError("pair_ids must be a nonempty list or tuple.")
    parsed = tuple(
        _pair_id(pair_id, name=f"pair_ids[{index}]")
        for index, pair_id in enumerate(pair_ids)
    )
    speed = _angular_speed(shield_angular_speed_rad_s)
    total = 0.0
    for target in parsed:
        total += shield_pair_transition_time_s(
            current,
            target,
            shield_angular_speed_rad_s=speed,
        )
        current = target
    return total


__all__ = [
    "DEFAULT_SHIELD_ANGULAR_SPEED_RAD_S",
    "shield_pair_transition_time_s",
    "shield_program_actuation_time_s",
]
