#!/usr/bin/env python3
"""
Passive follower joint reader: poll MECHANICAL_POSITION on all 16 RobStride motors.

No teleop, no MIT command stream, no ``follower_qpos_reader`` / MIT fallbacks — only
register reads so you can move the arm by hand and confirm encoders track.

CAN layout matches ``move_actuators.py`` / ``direct_teleop.py`` (7+1 per arm):
  left  ``can1`` ids 1,3,5,7,9,11,13,15 → indices 0..7
  right ``can0`` ids 2,4,6,8,10,12,14,16 → indices 8..15

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

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from move_actuators import (  # noqa: E402
    ARM_DOF,
    LEFT_CAN,
    LEFT_ROBSTRIDE_IDS,
    MOTOR_DIRECTION,
    MOTOR_MODEL_MAP,
    MOTOR_SOFTWARE_ZERO,
    NUM_JOINTS,
    RIGHT_CAN,
    RIGHT_ROBSTRIDE_IDS,
    _normalize_near_zero,
)

# L0..L6 + Lg, R0..R6 + Rg  (L5/L6 = wrist_roll/yaw motors 11/13; R5/R6 = 12/14)
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


def _repo_root() -> Path:
    """This file: <repo>/scripts/read_follower_joints.py"""
    p = Path(__file__).resolve().parent.parent
    if (p / "robstride_control").is_dir() or (p / "pyproject.toml").is_file():
        return p
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
    motors_cfg = {
        name: Motor(id=mid, model=MOTOR_MODEL_MAP.get(mid, "rs-02"))
        for mid, name in zip(motor_ids, motor_names)
    }
    calibration = {name: {"direction": 1, "homing_offset": 0.0} for name in motor_names}
    bus = RobstrideBus(can_channel, motors_cfg, calibration)
    bus.connect(handshake=True)
    for name in motor_names:
        bus.enable(name)
        time.sleep(0.05)
    return bus, list(zip(motor_names, motor_ids))


def _read_mechanical_16(
    left_bus,
    right_bus,
    left_motors: list[tuple[str, int]],
    right_motors: list[tuple[str, int]],
    parameter_type,
) -> np.ndarray:
    """Single-shot MECHANICAL_POSITION; failed joints stay 0.0 (no MIT / last-good).

    Applies the same ``MOTOR_DIRECTION`` + software-zero unwrap as ``move_actuators``
    so printed values match teleop / ``ActuatorController.read_joints``.
    """
    out = np.zeros(NUM_JOINTS, dtype=np.float64)

    def _one(bus, name: str, mid: int) -> float:
        raw = float(bus.read(name, parameter_type.MECHANICAL_POSITION))
        directed = raw * MOTOR_DIRECTION.get(mid, 1.0)
        return _normalize_near_zero(directed, MOTOR_SOFTWARE_ZERO.get(mid, 0.0))

    for i, (name, mid) in enumerate(left_motors):
        if left_bus is None:
            continue
        try:
            out[i] = _one(left_bus, name, mid)
        except Exception:
            pass
    for i, (name, mid) in enumerate(right_motors):
        if right_bus is None:
            continue
        try:
            out[ARM_DOF + i] = _one(right_bus, name, mid)
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
        description=(
            f"Read-only MECHANICAL_POSITION polling for {NUM_JOINTS} RobStride arm joints "
            "(7+1 per arm; no teleop / no MIT commands)."
        )
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
        help=f"line: one array per sample; table: per-joint; csv: t,sample,q0..q{NUM_JOINTS - 1}",
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
    print(
        f"Opening {LEFT_CAN} left={LEFT_ROBSTRIDE_IDS} and "
        f"{RIGHT_CAN} right={RIGHT_ROBSTRIDE_IDS} ({NUM_JOINTS} joints)...",
        flush=True,
    )

    left_bus, left_motors = _open_bus(
        RobstrideBus, Motor, ParameterType, LEFT_CAN, LEFT_ROBSTRIDE_IDS
    )
    right_bus, right_motors = _open_bus(
        RobstrideBus, Motor, ParameterType, RIGHT_CAN, RIGHT_ROBSTRIDE_IDS
    )
    if not left_bus and not right_bus:
        print("No buses connected.", file=sys.stderr)
        raise SystemExit(1)

    if not args.keep_torque_enabled:
        print(
            "Disabling motor torque for passive hand motion (use --keep-torque-enabled to skip).",
            flush=True,
        )
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
        f"Reading MECHANICAL_POSITION at ~{args.rate_hz} Hz ({NUM_JOINTS}D). "
        "Move joints by hand; Ctrl+C to exit.",
        flush=True,
    )
    if args.format == "csv":
        header = "t_unix,sample," + ",".join(
            f"qpos_{j}_{_LABELS_16[j]}" for j in range(NUM_JOINTS)
        )
        print(header, flush=True)

    n = 0
    try:
        while True:
            t0 = time.monotonic()
            q = _read_mechanical_16(
                left_bus, right_bus, left_motors, right_motors, ParameterType
            )
            n += 1
            if n % pe == 0:
                if args.format == "line":
                    inner = ", ".join(f"{float(q[j]):.{prec}f}" for j in range(NUM_JOINTS))
                    print(f"[{n:6d}] qpos = np.array([{inner}])  # len={NUM_JOINTS}", flush=True)
                elif args.format == "table":
                    print(f"\n--- sample={n} t={time.time():.3f} ---", flush=True)
                    for j in range(NUM_JOINTS):
                        mid = LEFT_ROBSTRIDE_IDS[j] if j < ARM_DOF else RIGHT_ROBSTRIDE_IDS[j - ARM_DOF]
                        print(
                            f"  [{j:2d}] {_LABELS_16[j]:4s}  id={mid:2d}  {float(q[j]):.{prec}f}",
                            flush=True,
                        )
                else:
                    ts = time.time()
                    row = [f"{ts:.6f}", str(n)] + [f"{float(q[j]):.6f}" for j in range(NUM_JOINTS)]
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
