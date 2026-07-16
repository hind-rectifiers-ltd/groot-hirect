#!/usr/bin/env python3
"""
Live preview to verify ``record/camera_ports.json`` maps HEAD / LEFT WRIST / RIGHT WRIST correctly.

Usage:
  uv run python record/preview_three_cameras.py

Discover USB port ids (when building camera_ports.json):
  uv run python record/preview_three_cameras.py --preview-all-cameras

Press Esc in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from usb_cameras import add_three_camera_cli_args, handle_camera_list_flags


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preview 3 cameras with role labels (verify camera_ports.json mapping)",
    )
    add_three_camera_cli_args(p)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--image-width", type=int, default=640)
    args = p.parse_args()
    if handle_camera_list_flags(args):
        return
    p.error("Use --preview-cameras (default when no other flag) or --preview-all-cameras")


if __name__ == "__main__":
    # Default to mapping preview when run as a script with no extra flags.
    if len(sys.argv) == 1:
        sys.argv.append("--preview-cameras")
    main()
