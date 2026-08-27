#!/usr/bin/env python3
"""
Passive follower joint reader: poll joints via ``ActuatorController.read_joints``.

Uses the same pipeline as record / teleop / safety (start-pose logical frame).

Default: **read-only connect** — opens CAN, does **not** enable torque / MODE / MIT
hold, so motors should not jerk. Use ``--keep-torque-enabled`` only if you need
the motors enabled (they will hold start pose).

Example:
  uv run python scripts/read_follower_joints.py --rate-hz 10
  uv run python scripts/read_follower_joints.py --format csv --rate-hz 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from move_actuators import (  # noqa: E402
    JOINT_MOTOR_IDS,
    NUM_JOINTS,
    ActuatorController,
)

# L0..L6 + Lg, R0..R6 + Rg
_LABELS_16 = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "Lg",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "Rg",
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            f"Read-only joint polling for {NUM_JOINTS} RobStride arm joints "
            "(same ActuatorController.read_joints path as record)."
        )
    )
    p.add_argument("--rate-hz", type=float, default=15.0, help="Target loop rate (sleep-paced)")
    p.add_argument(
        "--keep-torque-enabled",
        action="store_true",
        help="Enable torque and hold start pose (default: pure read-only, no enable).",
    )
    p.add_argument("--precision", type=int, default=4, help="Decimals for line/table output")
    p.add_argument(
        "--format",
        choices=("line", "table", "csv"),
        default="line",
        help=f"line: one array per sample; table: per-joint; csv: t,sample,q0..q{NUM_JOINTS - 1}",
    )
    p.add_argument("--print-every", type=int, default=1, help="Print every N samples")
    p.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Median sample count passed to read_joints (record uses 2 at 30 Hz)",
    )
    args = p.parse_args()

    arm = ActuatorController(
        safety_enabled=False,
        parallel_bus_reads=True,
        read_max_retries=4,
        enable_torque=bool(args.keep_torque_enabled),
    )
    arm.connect()
    if not args.keep_torque_enabled:
        print("Read-only mode: motors not enabled (no torque / no MIT hold).", flush=True)

    period = 1.0 / max(float(args.rate_hz), 0.25)
    prec = max(0, int(args.precision))
    pe = max(1, int(args.print_every))
    samples = max(1, int(args.samples))
    n = 0
    try:
        if args.format == "csv":
            hdr = "t,sample," + ",".join(f"q{i}" for i in range(NUM_JOINTS))
            print(hdr, flush=True)
        while True:
            t0 = time.monotonic()
            q = arm.read_joints(samples=samples)
            n += 1
            if n % pe == 0:
                if args.format == "line":
                    print(
                        f"--- sample={n} t={time.time():.3f} ---\n  "
                        + np.array2string(q, precision=prec, suppress_small=False),
                        flush=True,
                    )
                elif args.format == "csv":
                    print(
                        f"{time.time():.6f},{n},"
                        + ",".join(f"{float(v):.{prec}f}" for v in q),
                        flush=True,
                    )
                else:
                    print(f"\n--- sample={n} t={time.time():.3f} ---", flush=True)
                    for i, mid in enumerate(JOINT_MOTOR_IDS):
                        lab = _LABELS_16[i] if i < len(_LABELS_16) else f"j{i}"
                        print(
                            f"  [{i:2d}] {lab:4s}  id={mid:2d}  {float(q[i]):+.{prec}f}",
                            flush=True,
                        )
            elapsed = time.monotonic() - t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        try:
            print(arm.read_stats_line(), flush=True)
        except Exception:
            pass
        arm.disconnect(send_zero=False)


if __name__ == "__main__":
    main()
