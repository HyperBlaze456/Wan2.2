"""Simple smoke tests for `wan.flax.attention`.

Run with:
    python -m wan.flax.test_flash_attention
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from .attention import flash_attention


def _reference_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    mask: jnp.ndarray,
    scale_multiplier: float = 1.0,
) -> jnp.ndarray:
    """Float32 reference attention used to validate helper outputs."""
    q32 = jnp.asarray(q, dtype=jnp.float32) * scale_multiplier
    k32 = jnp.asarray(k, dtype=jnp.float32)
    v32 = jnp.asarray(v, dtype=jnp.float32)

    q_heads = jnp.transpose(q32, (0, 2, 1, 3))  # (B, H, T, C)
    k_heads = jnp.transpose(k32, (0, 2, 1, 3))  # (B, H, S, C)
    v_heads = jnp.transpose(v32, (0, 2, 1, 3))

    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    logits = jnp.einsum("bhtd,bhsd->bhts", q_heads * scale, k_heads)
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)

    weights = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("bhts,bhsd->bhtd", weights, v_heads)
    return jnp.transpose(attended, (0, 2, 1, 3))  # (B, T, H, C)


def _make_length_mask(
    batch: int,
    q_len: int,
    k_len: int,
    num_heads: int,
    *,
    q_lens: jnp.ndarray | None = None,
    k_lens: jnp.ndarray | None = None,
    causal: bool = False,
) -> jnp.ndarray:
    mask = jnp.ones((batch, num_heads, q_len, k_len), dtype=jnp.bool_)
    if q_lens is not None:
        q_lens = jnp.asarray(q_lens, dtype=jnp.int32)
        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        mask = mask & (q_idx[None, None, :, None] < q_lens[:, None, None, None])
    if k_lens is not None:
        k_lens = jnp.asarray(k_lens, dtype=jnp.int32)
        k_idx = jnp.arange(k_len, dtype=jnp.int32)
        mask = mask & (k_idx[None, None, None, :] < k_lens[:, None, None, None])
    if causal:
        q_idx = jnp.arange(q_len, dtype=jnp.int32)[:, None]
        k_idx = jnp.arange(k_len, dtype=jnp.int32)[None, :]
        causal_mask = k_idx <= q_idx
        mask = mask & causal_mask[None, None, :, :]
    return mask


def _run_case(
    *,
    key: jax.Array,
    batch: int = 2,
    q_len: int = 6,
    k_len: int = 6,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    value_dim: int | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    q_lens: jnp.ndarray | None = None,
    k_lens: jnp.ndarray | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
) -> None:
    if value_dim is None:
        value_dim = head_dim

    keys = random.split(key, 3)
    q = random.normal(keys[0], (batch, q_len, num_q_heads, head_dim), dtype=dtype)
    k = random.normal(keys[1], (batch, k_len, num_kv_heads, head_dim), dtype=dtype)
    v = random.normal(keys[2], (batch, k_len, num_kv_heads, value_dim), dtype=dtype)

    scale_multiplier = 1.0
    if q_scale is not None:
        scale_multiplier *= q_scale
    if softmax_scale is not None:
        scale_multiplier *= softmax_scale

    mask = _make_length_mask(
        batch,
        q_len,
        k_len,
        num_q_heads,
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
    )

    out = flash_attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        dtype=dtype,
    )

    ref = _reference_attention(
        q,
        jnp.repeat(k, repeats=num_q_heads // num_kv_heads, axis=2),
        jnp.repeat(v, repeats=num_q_heads // num_kv_heads, axis=2),
        mask=mask,
        scale_multiplier=scale_multiplier,
    )

    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        rtol=3e-3,
        atol=3e-3,
    )


def main() -> None:
    print("Running wan.flax.attention smoke tests...")
    key = random.key(0)

    cases = [
        {},
        {"causal": True},
        {"q_lens": jnp.array([6, 4], dtype=jnp.int32), "k_lens": jnp.array([5, 3], dtype=jnp.int32)},
        {"softmax_scale": 0.5, "q_scale": 1.1},
    ]

    for idx, case_kwargs in enumerate(cases, start=1):
        print(f"- case {idx}: {case_kwargs if case_kwargs else 'baseline'}")
        _run_case(key=key, **case_kwargs)
        key, _ = random.split(key)

    print("All wan.flax.attention smoke tests passed.")


if __name__ == "__main__":
    main()
