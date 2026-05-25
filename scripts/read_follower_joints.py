#!/usr/bin/env python3
"""
Passive follower joint reader: poll MECHANICAL_POSITION on all 12 RobStride motors.

No teleop, no MIT command stream, no ``follower_qpos_reader`` / MIT fallbacks — only
register reads so you can move the arm by hand and confirm encoders track.

CAN layout matches ``direct_teleop.py``: left ``can1`` ids 1,3,5,7,9,11 → indices 0..5;
right ``can0`` ids 2,4,6,8,10,12 → indices 6..11.

Default: connect, enable briefly, then **disable** all motors so you can backdrive by hand.
If reads fail after disable, try ``--keep-torque-enabled`` (motors may hold position).

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

# --- Same arm layout as direct_teleop.py (duplicated to avoid importing teleop) ---
LEFT_ROBSTRIDE_IDS = [1, 3, 5, 7, 9, 11]
RIGHT_ROBSTRIDE_IDS = [2, 4, 6, 8, 10, 12]
LEFT_CAN = "can1"
RIGHT_CAN = "can0"

MOTOR_MODEL_MAP = {
    1: "rs-03",
    2: "rs-03",
    3: "rs-03",
    4: "rs-03",
    5: "rs-06",
    6: "rs-06",
    7: "rs-06",
    8: "rs-06",
    9: "rs-02",
    10: "rs-02",
    11: "rs-02",
    12: "rs-02",
}

_LABELS_12 = ("L0", "L1", "L2", "L3", "L4", "Lg", "R0", "R1", "R2", "R3", "R4", "Rg")


def _repo_root() -> Path:
    """This file: <repo>/scripts/read_follower_joints.py"""
    p = Path(__file__).resolve().parent.parent
    if (p / "robstride_control").is_dir() or (p / "pyproject.toml").is_file():
        return p
    # Fallback: parent of scripts
    return Path(__file__).resolve().parents[1]


def _ensure_import_paths(repo: Path) -> None:
    src = repo / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for root in (repo, Path.cwd(), Path.cwd().parent):
        rd = root / "robstride_control"
        if rd.is_dir() and str(rd) not in sys.path:
            sys.path.insert(0, str(rd))


def _open_bus(RobstrideBus, Motor, ParameterType, can_channel: str, motor_ids: list[int]):
    if not motor_ids:
        return None, []
    motor_names = [f"motor_{mid}" for mid in motor_ids]
    motors_cfg = {name: Motor(id=mid, model=MOTOR_MODEL_MAP.get(mid, "rs-02")) for mid, name in zip(motor_ids, motor_names)}
    calibration = {name: {"direction": 1, "homing_offset": 0.0} for name in motor_names}
    bus = RobstrideBus(can_channel, motors_cfg, calibration)
    bus.connect(handshake=True)
    for name in motor_names:
        bus.enable(name)
        time.sleep(0.05)
    return bus, list(zip(motor_names, motor_ids))


def _read_mechanical_12(
    left_bus,
    right_bus,
    left_motors: list[tuple[str, int]],
    right_motors: list[tuple[str, int]],
    parameter_type,
) -> np.ndarray:
    """Single-shot MECHANICAL_POSITION only; failed joints stay 0.0 (no MIT / last-good)."""
    out = np.zeros(12, dtype=np.float64)
    for i, (name, _mid) in enumerate(left_motors):
        if left_bus is None:
            continue
        try:
            out[i] = float(left_bus.read(name, parameter_type.MECHANICAL_POSITION))
        except Exception:
            pass
    for i, (name, _mid) in enumerate(right_motors):
        if right_bus is None:
            continue
        try:
            out[6 + i] = float(right_bus.read(name, parameter_type.MECHANICAL_POSITION))
        except Exception:
            pass
    return out


def _disconnect(left_bus, left_motors, right_bus, right_motors) -> None:
    for bus, motors in ((left_bus, left_motors), (right_bus, right_motors)):
        if bus is None or not motors:
            continue
        try:
            for motor_name, _ in motors:
                try:
                    bus.disable(motor_name)
                except Exception:
                    pass
            bus.disconnect()
        except Exception as e:
            print(f"Disconnect warning: {e}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Read-only MECHANICAL_POSITION polling for 12 RobStride arm joints (no teleop / no MIT commands)."
    )
    p.add_argument("--rate-hz", type=float, default=15.0, help="Target loop rate (sleep-paced)")
    p.add_argument(
        "--keep-torque-enabled",
        action="store_true",
        help="Do not call disable() after connect; motors may resist hand motion but reads often work.",
    )
    p.add_argument("--precision", type=int, default=4, help="Decimals for line/table output")
    p.add_argument(
        "--format",
        choices=("line", "table", "csv"),
        default="line",
        help="line: one array per sample; table: per-joint; csv: t,sample,q0..q11",
    )
    p.add_argument("--print-every", type=int, default=1, help="Print every N samples")
    args = p.parse_args()

    repo = _repo_root()
    _ensure_import_paths(repo)

    try:
        from robstride_dynamics import Motor, ParameterType, RobstrideBus
    except Exception:
        from robstride_dynamics.bus import Motor, RobstrideBus
        from robstride_dynamics.protocol import ParameterType

    print(f"Repo root: {repo}", flush=True)
    print(f"Opening {LEFT_CAN} (left) and {RIGHT_CAN} (right)...", flush=True)

    left_bus, left_motors = _open_bus(RobstrideBus, Motor, ParameterType, LEFT_CAN, LEFT_ROBSTRIDE_IDS)
    right_bus, right_motors = _open_bus(RobstrideBus, Motor, ParameterType, RIGHT_CAN, RIGHT_ROBSTRIDE_IDS)
    if not left_bus and not right_bus:
        print("No buses connected.", file=sys.stderr)
        raise SystemExit(1)

    if not args.keep_torque_enabled:
        print("Disabling motor torque for passive hand motion (use --keep-torque-enabled to skip).", flush=True)
        for bus, motors in ((left_bus, left_motors), (right_bus, right_motors)):
            if bus is None:
                continue
            for motor_name, _ in motors:
                try:
                    bus.disable(motor_name)
                except Exception as e:
                    print(f"  disable {motor_name}: {e}", file=sys.stderr)
                time.sleep(0.02)

    period = 1.0 / max(float(args.rate_hz), 0.25)
    prec = max(0, int(args.precision))
    pe = max(1, int(args.print_every))

    print(
        f"Reading MECHANICAL_POSITION at ~{args.rate_hz} Hz (mechanical only). "
        "Move joints by hand; Ctrl+C to exit.",
        flush=True,
    )
    if args.format == "csv":
        print("t_unix,sample," + ",".join(f"qpos_{j}_{_LABELS_12[j]}" for j in range(12)), flush=True)

    n = 0
    try:
        while True:
            t0 = time.monotonic()
            q = _read_mechanical_12(left_bus, right_bus, left_motors, right_motors, ParameterType)
            n += 1
            if n % pe == 0:
                if args.format == "line":
                    inner = ", ".join(f"{float(q[j]):.{prec}f}" for j in range(12))
                    print(f"[{n:6d}] qpos = np.array([{inner}])", flush=True)
                elif args.format == "table":
                    print(f"\n--- sample={n} t={time.time():.3f} ---", flush=True)
                    for j in range(12):
                        print(f"  [{j:2d}] {_LABELS_12[j]:4s}  {float(q[j]):.{prec}f}", flush=True)
                else:
                    ts = time.time()
                    row = [f"{ts:.6f}", str(n)] + [f"{float(q[j]):.6f}" for j in range(12)]
                    print(",".join(row), flush=True)

            elapsed = time.monotonic() - t0
            slp = period - elapsed
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        _disconnect(left_bus, left_motors, right_bus, right_motors)
        print("Buses disconnected.", flush=True)


if __name__ == "__main__":
    main()
