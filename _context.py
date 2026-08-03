"""Shared helpers for the models' `context` field (context window sizes)."""

from __future__ import annotations

# Advertised window sizes. Sources report the raw token count instead (262144,
# 131072, 1048576), which reads as an odd label unless snapped back to the size
# the creator actually markets.
CANONICAL_WINDOWS = [
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    100_000,
    128_000,
    192_000,
    200_000,
    # MiniMax markets 204800 tokens as "204k", so it is its own rung rather
    # than noise around 200k.
    204_800,
    256_000,
    512_000,
    1_000_000,
    2_000_000,
    4_000_000,
    10_000_000,
]
SNAP_TOLERANCE = 0.05


def snap_context_tokens(tokens: int) -> int:
    """Snap a token count to the nearest advertised size (262144 -> 256000)."""
    nearest = min(CANONICAL_WINDOWS, key=lambda target: abs(tokens - target) / target)
    if abs(tokens - nearest) / nearest <= SNAP_TOLERANCE:
        return nearest
    return tokens


def format_context_tokens(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    if tokens >= 1_000:
        return f"{tokens // 1_000}k"
    return str(tokens)
