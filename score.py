#!/usr/bin/env python3
"""Backward-compatible wrapper for edit.py."""

from __future__ import annotations

from edit import main


if __name__ == "__main__":
    raise SystemExit(main())
