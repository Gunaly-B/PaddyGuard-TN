"""Backward-compatible entry point for building advisory CSV files."""
from __future__ import annotations

from scripts.build_advisory_data import build


if __name__ == "__main__":
    result = build()
    for key, value in result.items():
        print(f"{key}: {value}")
