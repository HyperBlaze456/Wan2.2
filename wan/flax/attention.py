from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

try:
    from jax.experimental.pallas.ops.tpu import splash_attention as splash
    _SPLASH_AVAILABLE = True
except ImportError:
    try:
        from jax.experimental.pallas.ops.gpu import splash_attention as splash
        _SPLASH_AVAILABLE = True
    except ImportError:
        splash = None
        _SPLASH_AVAILABLE = False

__all__ = ["flash_attention", "attention"]

_HALF_DTYPES = (jnp.bfloat16, jnp.float16)

# Dimension legend:
#   1: batch (B)
#   2: query length (T)
#   3: num query heads (Hq)
#   4: head channel size (Cq)
#   5: key length (S)
#   6: num key/value heads (Hkv)
#   7: value channel size (Cv)


def _broadcast_mask(
    mask: jnp.ndarray,
    batch_size: int,
    num_heads: int,
    q_len: int,
    k_len: int,
) -> jnp.ndarray:
    # Broadcast per-head mask to (1 B, 3 Hq, 2 T, 5 S).
    mask = jnp.broadcast_to(mask[:, None, :, :], (batch_size, num_heads, q_len, k_len))
    return mask.astype(jnp.bool_)


def _build_attention_mask(
    batch_size: int,
    num_heads: int,
    q_len: int,
    k_len: int,
    q_lens: Optional[jnp.ndarray],
    k_lens: Optional[jnp.ndarray],
    causal: bool,
    window_size: Tuple[int, int],
) -> jnp.ndarray:
    # base[1 B, 2 T, 5 S] tracks allowed query-key connections before head broadcast.
    base = jnp.ones((batch_size, q_len, k_len), dtype=jnp.bool_)

    if q_lens is not None:
        q_lens = jnp.asarray(q_lens, dtype=jnp.int32)
        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        # Mask out query positions (axis 2) beyond per-sample lengths.
        base = base & (q_idx[None, :] < q_lens[:, None])[:, :, None]

    if k_lens is not None:
        k_lens = jnp.asarray(k_lens, dtype=jnp.int32)
        k_idx = jnp.arange(k_len, dtype=jnp.int32)
        # Mask out key positions (axis 3) beyond per-sample lengths.
        base = base & (k_idx[None, :] < k_lens[:, None])[:, None, :]

    if causal:
        q_idx = jnp.arange(q_len, dtype=jnp.int32)[:, None]
        k_idx = jnp.arange(k_len, dtype=jnp.int32)[None, :]
        # Enforce k <= q so tokens only attend to earlier (or same) positions.
        base = base & (k_idx <= q_idx)[None, :, :]

    left, right = window_size
    if left != -1 or right != -1:
        left = None if left < 0 else left
        right = None if right < 0 else right
        q_idx = jnp.arange(q_len, dtype=jnp.int32)[:, None]
        k_idx = jnp.arange(k_len, dtype=jnp.int32)[None, :]
        # Local window mask narrows attention bandwidth around the diagonal.
        window_mask = jnp.ones((q_len, k_len), dtype=jnp.bool_)
        if left is not None:
            window_mask = window_mask & (k_idx >= q_idx - left)
        if right is not None:
            window_mask = window_mask & (k_idx <= q_idx + right)
        base = base & window_mask[None, :, :]

    return _broadcast_mask(base, batch_size, num_heads, q_len, k_len)


def _repeat_heads(tensor: jnp.ndarray, repeats: int) -> jnp.ndarray:
    if repeats == 1:
        return tensor
    return jnp.repeat(tensor, repeats=repeats, axis=1)


def _flash_attention_with_splash(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    dtype: jnp.dtype,
    scale_multiplier,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    # q: [1 B, 2 T, 3 Hq, 4 Cq], k: [1 B, 5 S, 6 Hkv, 4 Cq], v: [1 B, 5 S, 6 Hkv, 7 Cv]
    if dtype not in _HALF_DTYPES:
        raise ValueError(f"Splash attention requires one of {_HALF_DTYPES}, got {dtype}.")

    batch, q_len, num_q_heads, head_dim = q.shape
    _, k_len, num_kv_heads, _ = k.shape
    value_dim = v.shape[-1]

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Number of query heads must be divisible by number of key/value heads.")

    group_size = num_q_heads // num_kv_heads
    output_dtype = q.dtype

    scale_multiplier = jnp.asarray(scale_multiplier, dtype=jnp.float32)

    q_half = jnp.asarray(q, dtype=dtype) * scale_multiplier
    k_half = jnp.asarray(k, dtype=dtype)
    v_half = jnp.asarray(v, dtype=dtype)

    # Flatten heads so Splash works with per-head sequences: (B*Hq, T, Cq/Cv).
    q_heads = jnp.transpose(q_half, (0, 2, 1, 3)).reshape(batch * num_q_heads, q_len, head_dim)
    k_heads = _repeat_heads(jnp.transpose(k_half, (0, 2, 1, 3)), group_size).reshape(
        batch * num_q_heads, k_len, head_dim
    )
    v_heads = _repeat_heads(jnp.transpose(v_half, (0, 2, 1, 3)), group_size).reshape(
        batch * num_q_heads, k_len, value_dim
    )

    # Splash expects boolean mask per head: (B*Hq, T, S).
    mask_per_head = mask.reshape(batch * num_q_heads, q_len, k_len)

    kernel = splash.make_splash_mha_single_device(mask_per_head)
    attn = kernel(q_heads, k_heads, v_heads)

    out = attn.reshape(batch, num_q_heads, q_len, value_dim)
    out = jnp.transpose(out, (0, 2, 1, 3))
    return out.astype(output_dtype)


def _scaled_dot_product_attention_fallback(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    mask: jnp.ndarray,
    scale_multiplier,
) -> jnp.ndarray:
    # q/k/v follow the numbering above; fallback runs in float32 for stability.
    batch, q_len, num_q_heads, head_dim = q.shape
    _, k_len, num_kv_heads, _ = k.shape

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Number of query heads must be divisible by number of key/value heads.")

    group_size = num_q_heads // num_kv_heads
    output_dtype = q.dtype

    scale_multiplier = jnp.asarray(scale_multiplier, dtype=jnp.float32)

    q_f32 = jnp.asarray(q, dtype=jnp.float32) * scale_multiplier
    k_f32 = jnp.asarray(k, dtype=jnp.float32)
    v_f32 = jnp.asarray(v, dtype=jnp.float32)

    q_heads = jnp.transpose(q_f32, (0, 2, 1, 3))
    k_heads = _repeat_heads(jnp.transpose(k_f32, (0, 2, 1, 3)), group_size)
    v_heads = _repeat_heads(jnp.transpose(v_f32, (0, 2, 1, 3)), group_size)

    scale = 1.0 / math.sqrt(head_dim)
    logits = jnp.einsum("bnqd,bnkd->bnqk", q_heads * scale, k_heads)

    mask_value = jnp.finfo(jnp.float32).min
    logits = jnp.where(mask, logits, mask_value)

    weights = jax.nn.softmax(logits, axis=-1)

    attn = jnp.einsum("bnqk,bnkd->bnqd", weights, v_heads)

    out = jnp.transpose(attn, (0, 2, 1, 3))
    return out.astype(output_dtype)


def flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    q_lens: Optional[jnp.ndarray] = None,
    k_lens: Optional[jnp.ndarray] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = True,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    # Inputs follow [1 B, 2 T, 3 Hq, 4 Cq] and [1 B, 5 S, 6 Hkv, 4 Cq / 7 Cv].
    del deterministic

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to have shape [batch, length, heads, features].")

    if k.shape[:2] != v.shape[:2] or q.shape[0] != k.shape[0]:
        raise ValueError("Inconsistent shapes between q, k, and v.")

    batch, q_len, num_q_heads, _ = q.shape
    k_len = k.shape[1]

    mask = _build_attention_mask(
        batch,
        num_q_heads,
        q_len,
        k_len,
        q_lens,
        k_lens,
        causal,
        window_size,
    )
    # mask expands to [1 B, 3 Hq, 2 T, 5 S] once broadcast.

    scale_multiplier = 1.0
    if q_scale is not None:
        scale_multiplier = scale_multiplier * jnp.asarray(q_scale, dtype=jnp.float32)
    if softmax_scale is not None:
        scale_multiplier = scale_multiplier * jnp.asarray(softmax_scale, dtype=jnp.float32)

    if dropout_p:
        warnings.warn(
            "Dropout is not supported in this attention helper; falling back to standard attention.",
            stacklevel=2,
        )
        return _scaled_dot_product_attention_fallback(
            q,
            k,
            v,
            mask=mask,
            scale_multiplier=scale_multiplier,
        )

    if not _SPLASH_AVAILABLE:
        warnings.warn(
            "Pallas splash attention is not available; using scaled dot product attention.",
            stacklevel=2,
        )
        return _scaled_dot_product_attention_fallback(
            q,
            k,
            v,
            mask=mask,
            scale_multiplier=scale_multiplier,
        )

    if dtype not in _HALF_DTYPES:
        warnings.warn(
            f"Splash attention requires half precision dtypes {_HALF_DTYPES}; using fallback.",
            stacklevel=2,
        )
        return _scaled_dot_product_attention_fallback(
            q,
            k,
            v,
            mask=mask,
            scale_multiplier=scale_multiplier,
        )

    try:
        return _flash_attention_with_splash(
            q,
            k,
            v,
            dtype=dtype,
            scale_multiplier=scale_multiplier,
            mask=mask,
        )
    except Exception as exc:
        warnings.warn(
            f"Pallas splash attention fallback triggered due to: {exc}",
            stacklevel=2,
        )
        return _scaled_dot_product_attention_fallback(
            q,
            k,
            v,
            mask=mask,
            scale_multiplier=scale_multiplier,
        )


def attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    q_lens: Optional[jnp.ndarray] = None,
    k_lens: Optional[jnp.ndarray] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = True,
    dtype: jnp.dtype = jnp.bfloat16,
    fa_version: Optional[int] = None,
) -> jnp.ndarray:
    del fa_version
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
    )
