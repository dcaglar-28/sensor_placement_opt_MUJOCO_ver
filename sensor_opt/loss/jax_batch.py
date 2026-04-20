"""
JAX-friendly batched loss computation.

This module keeps a strict separation between:
  - Python/OO objects (SensorConfig, dataclasses)
  - batched numeric arrays (JAX / NumPy)

So you can evaluate many candidates at once (e.g. CMA population) without
changing the rest of the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None


def loss_from_metrics_batch(
    collision_rate: Any,
    blind_spot_fraction: Any,
    cost_usd: Any,
    n_active_sensors: Any,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    max_cost_usd: float = 10_000.0,
) -> Any:
    """
    Vectorized loss: accepts arrays shaped (B,) and returns (B,).

    Works with:
      - JAX arrays (jit/vmap friendly) when JAX is installed
      - NumPy arrays otherwise
    """
    xp = jnp if jnp is not None else np
    collision_rate = xp.asarray(collision_rate)
    blind_spot_fraction = xp.asarray(blind_spot_fraction)
    cost_usd = xp.asarray(cost_usd)
    n_active_sensors = xp.asarray(n_active_sensors)

    collision_term = alpha * xp.clip(collision_rate, 0.0, 1.0)
    blind_term = beta * xp.clip(blind_spot_fraction, 0.0, 1.0)
    cost_penalty = xp.clip(cost_usd / max_cost_usd, 0.0, 1.0)
    cost_term = gamma * cost_penalty
    total = collision_term + blind_term + cost_term

    # Match scalar behavior: empty config => loss 1.0
    total = xp.where(n_active_sensors <= 0, xp.asarray(1.0, dtype=total.dtype), total)
    return xp.clip(total, 0.0, 1.0)


def jit_loss_from_metrics_batch():
    """
    Convenience: returns a jitted version if JAX is available.
    """
    if jax is None:
        raise ImportError("JAX is not installed; cannot jit.")
    return jax.jit(loss_from_metrics_batch)

