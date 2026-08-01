"""Define reproducible, order-independent random streams for simulation work."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

RNG_PROVENANCE_SCHEMA_VERSION = 1
RNG_BIT_GENERATOR = "PCG64"
_RNG_NAMED_STREAM_DOMAIN = 0x52534E53
_RNG_MAX_ROOT_SEED = (1 << 128) - 1


def normalize_random_seed(value: object) -> int:
    """Return one canonical non-negative 128-bit simulation root seed."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError("Random seed must be an integer.")
    seed = int(value)
    if seed < 0 or seed > _RNG_MAX_ROOT_SEED:
        raise ValueError("Random seed must be in [0, 2**128 - 1].")
    return seed


def named_stream_spawn_key(*components: object) -> tuple[int, ...]:
    """Encode a stable named substream without using Python's process hash."""
    if not components:
        raise ValueError("A named random stream requires at least one component.")
    encoded_components = [str(component).encode("utf-8") for component in components]
    if any(not component for component in encoded_components):
        raise ValueError("Named random-stream components must be non-empty.")
    key: list[int] = [
        _RNG_NAMED_STREAM_DOMAIN,
        RNG_PROVENANCE_SCHEMA_VERSION,
        len(encoded_components),
    ]
    for component in encoded_components:
        key.extend((len(component), *(int(value) for value in component)))
    return tuple(key)


def named_seed_sequence(
    root_seed: int,
    *components: object,
) -> np.random.SeedSequence:
    """Derive one order-independent named stream from a logged root seed."""
    return np.random.SeedSequence(
        normalize_random_seed(root_seed),
        spawn_key=named_stream_spawn_key(*components),
    )


def named_random_generator(
    root_seed: int,
    *components: object,
) -> np.random.Generator:
    """Build a deterministic PCG64 generator for a named runtime operation."""
    return np.random.Generator(
        np.random.PCG64(named_seed_sequence(root_seed, *components))
    )


def named_stream_seed(
    root_seed: int,
    *components: object,
) -> int:
    """Return a deterministic unsigned 64-bit seed for a named runtime stream."""
    words = named_seed_sequence(root_seed, *components).generate_state(
        2,
        dtype=np.uint32,
    )
    return int(words[0]) | (int(words[1]) << 32)


def named_rng_provenance(
    root_seed: int,
    stream_names: Sequence[str],
) -> dict[str, Any]:
    """Describe named planning streams derived from one logged root seed."""
    seed = normalize_random_seed(root_seed)
    names = tuple(sorted({str(name) for name in stream_names}))
    if not names or any(not name for name in names):
        raise ValueError("Named RNG provenance requires non-empty stream names.")
    return {
        "schema_version": RNG_PROVENANCE_SCHEMA_VERSION,
        "root_seed": seed,
        "derivation_method": (
            "numpy_seedsequence_named_utf8_spawn_key_v1"
        ),
        "bit_generator": RNG_BIT_GENERATOR,
        "streams": {
            name: {
                "domain": name,
                "spawn_key": list(named_stream_spawn_key(name)),
                "derived_seed_u64": named_stream_seed(seed, name),
            }
            for name in names
        },
    }


__all__ = [
    "RNG_BIT_GENERATOR",
    "RNG_PROVENANCE_SCHEMA_VERSION",
    "named_random_generator",
    "named_rng_provenance",
    "named_seed_sequence",
    "named_stream_seed",
    "named_stream_spawn_key",
    "normalize_random_seed",
]
