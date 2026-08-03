"""Shared helpers for the models' `params` field (parameter counts).

Artificial Analysis is the primary source: its model pages carry both the total
and the active (MoE) parameter count. Hugging Face is the fallback for models AA
has no page for, but its API exposes only a total, so the "-A..." half of the
string stays a manual edit there.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any

HF_API_URL = "https://huggingface.co/api/models/{}"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"

# "117B", "1T-A32B", "68.5B-A3B", "270M", plus Gemma's effective size "E2B".
_PARAMS_RE = re.compile(
    r"^(E?[0-9]+(?:\.[0-9]+)?)([BMT])(?:-A([0-9]+(?:\.[0-9]+)?)([BMT]))?$"
)


def format_param_billions(value: Any) -> str:
    """Render a parameter count given in billions as the site's shorthand."""
    if value is None:
        return ""
    try:
        billions = float(value)
    except (TypeError, ValueError):
        return ""
    if billions <= 0:
        return ""

    if billions < 1:
        # Sub-billion counts read as millions at two significant digits, which
        # is how creators name those releases (0.268 -> 270M).
        millions = billions * 1000.0
        exponent = math.floor(math.log10(millions))
        return f"{int(round(millions, 1 - exponent))}M"

    if billions >= 1000:
        amount, unit, tolerance = billions / 1000.0, "T", 0.03
    else:
        # Kept tight in the B range: half-steps like 68.5B are advertised sizes,
        # not measurement noise, so they must survive.
        amount, unit, tolerance = billions, "B", 0.005

    # Both sources report measured counts, so they land just off the advertised
    # size (79.7 -> 80B, 1023 -> 1T); snap when the gap is within that noise.
    nearest = round(amount)
    if nearest and abs(amount - nearest) / nearest <= tolerance:
        amount = float(nearest)
    else:
        amount = round(amount, 1)

    if amount.is_integer():
        return f"{int(amount)}{unit}"
    return f"{amount:g}{unit}"


def format_params(total: Any, active: Any = None) -> str:
    """Combine total/active counts (in billions) into "235B-A22B" form."""
    total_text = format_param_billions(total)
    if not total_text:
        return ""
    active_text = format_param_billions(active)
    if active_text:
        return f"{total_text}-A{active_text}"
    return total_text


def normalize_params(value: Any) -> str | None:
    """Validate a params string, normalizing unit case ("117b" -> "117B")."""
    if not isinstance(value, str):
        return None
    raw = value.strip().upper().replace(" ", "")
    if not raw:
        return None
    match = _PARAMS_RE.match(raw)
    if not match:
        return None
    total, total_unit, active, active_unit = match.groups()
    text = f"{total}{total_unit}"
    if active:
        text += f"-A{active}{active_unit}"
    return text


def _hf_repo(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    match = re.match(r"^https?://huggingface\.co/([^/?#]+/[^/?#]+)", url.strip())
    return match.group(1) if match else ""


def _auth_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_hf_params(url: Any, timeout: int = 15) -> str | None:
    """Total parameter count for a Hugging Face repo, or None.

    Reads `safetensors.total` from the repo's API record. The API carries no
    active-parameter count, so MoE models come back as a total only ("117B").
    """
    repo = _hf_repo(url)
    if not repo:
        return None

    request = urllib.request.Request(HF_API_URL.format(repo), headers=_auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    safetensors = payload.get("safetensors")
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if not isinstance(total, (int, float)) or isinstance(total, bool) or total <= 0:
        return None

    return normalize_params(format_param_billions(total / 1_000_000_000))
