#!/usr/bin/env python3
"""
Direct local teleop: read leader Dynamixels and drive follower RobStride joints.

This merges joint_client.py + joint_server.py into one process on one machine.
No UDP/websocket is used.

Flow:
- Read leader present positions from Dynamixel motors (raw units).
- Use first sample as teleop zero (capture follower mechanical refs).
- Accumulate shortest-path deltas in leader raw units.
- Convert deltas to follower target radians and drive motors via ``ActuatorController``
  from ``move_actuators``.

Optional: ``--print-follower-qpos`` uses ``record/follower_qpos_reader.py`` (retries,
MIT status fallback, last-good per joint) like ``record/record_episodes_3cam.py``,
and by default polls encoders **before** each MIT write burst on print ticks to
reduce CAN contention. See ``--help`` for tuning flags.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Leader (Dynamixel) constants
# ---------------------------------------------------------------------------

SERVO_UNITS_PER_REV = 4096.0
RAD_PER_SERVO_UNIT = 2.0 * math.pi / SERVO_UNITS_PER_REV

GRIPPER_MOTION_SCALE = 5.0
GRIPPER_MOTOR_IDS = {11, 12}
GRIPPER_SERVO_INDICES = {10, 11}

# ---------------------------------------------------------------------------
# Re-export arm layout constants from move_actuators for callers that import
# direct_teleop (e.g. record_episodes_3cam, policy_client_3cam).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from move_actuators import (  # noqa: E402
    ActuatorController,
    LEFT_CAN,
    LEFT_ROBSTRIDE_IDS,
    MOTOR_KD,
    MOTOR_KP,
    MOTOR_MODEL_MAP,
    MOTOR_TORQUE_LIMIT,
    NUM_JOINTS,
    RAMP_DT_MAX_S as ROBSTRIDE_RAMP_DT_MAX_S,
    RAMP_MAX_SPEED_RAD_S as ROBSTRIDE_RAMP_MAX_SPEED_RAD_S,
    RIGHT_CAN,
    RIGHT_ROBSTRIDE_IDS,
    _open_bus,
    _ensure_robstride_on_path,
    _load_robstride,
)

# ---------------------------------------------------------------------------
# Helpers kept in this file (leader / teleop-specific logic)
# ---------------------------------------------------------------------------

def pad12(angles):
    a = [float(x) for x in angles]
    while len(a) < 12:
        a.append(0.0)
    return a[:12]


def shortest_delta_units(prev_u: float, curr_u: float, period: float = SERVO_UNITS_PER_REV) -> float:
    d = float(curr_u) - float(prev_u)
    p = period
    return (d + p / 2.0) % p - p / 2.0


def accum_units_to_target_delta_rad(accum_units: float, motor_id: int, servo_idx: int | None = None) -> float:
    delta_rad = accum_units * RAD_PER_SERVO_UNIT
    if motor_id in GRIPPER_MOTOR_IDS or servo_idx in GRIPPER_SERVO_INDICES:
        delta_rad *= GRIPPER_MOTION_SCALE
    if motor_id in {1, 2, 5, 6, 7, 8, 9, 10, 11, 12}:
        return -delta_rad
    return delta_rad


def ramp_toward(current: float, desired: float, max_step: float) -> float:
    err = float(desired) - float(current)
    if err > max_step:
        return float(current) + max_step
    if err < -max_step:
        return float(current) - max_step
    return float(desired)


def ensure_import_paths(project_root: Path) -> None:
    src = project_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    record_dir = project_root / "record"
    if record_dir.is_dir() and str(record_dir) not in sys.path:
        sys.path.insert(0, str(record_dir))

    _ensure_robstride_on_path()


def init_robstride_bus(RobstrideBus, Motor, ParameterType, can_channel, motor_ids):
    """Thin wrapper kept for callers that import this function directly."""
    return _open_bus(RobstrideBus, Motor, ParameterType, can_channel, motor_ids)


def get_joint_angles_from_motors(motors):
    return [float(m.getPresentPosition()) for m in motors]


# ---------------------------------------------------------------------------
# Logging helpers (--print-follower-qpos)
# ---------------------------------------------------------------------------

_FOLLOWER_JOINT_LABELS_12 = ("L0", "L1", "L2", "L3", "L4", "Lg", "R0", "R1", "R2", "R3", "R4", "Rg")


def ramped_cmd_to_action12(left_motors, right_motors, ramped_cmd: dict) -> np.ndarray:
    action = np.zeros(12, dtype=np.float64)
    for i, (name, _mid) in enumerate(left_motors):
        action[i] = float(ramped_cmd.get(name, 0.0))
    for i, (name, _mid) in enumerate(right_motors):
        action[6 + i] = float(ramped_cmd.get(name, 0.0))
    return action


def _format_joint_vector_line(name: str, row: np.ndarray, *, precision: int) -> str:
    r = np.asarray(row, dtype=np.float64).reshape(-1)
    inner = ", ".join(f"{float(v):.{precision}f}" for v in r)
    return f"{name} = np.array([{inner}])  # len={len(r)}"


def print_follower_qpos_action_block(*, loop_n: int, qpos12: np.ndarray, action12: np.ndarray, precision: int) -> None:
    print(f"\n--- [direct_teleop] loop={loop_n} ---")
    print(_format_joint_vector_line("qpos", qpos12, precision=precision))
    print(_format_joint_vector_line("action", action12, precision=precision))
    print("per_joint (index label qpos action):")
    for j in range(min(12, len(qpos12), len(action12))):
        lab = _FOLLOWER_JOINT_LABELS_12[j]
        print(f"  [{j:2d}] {lab:4s}  qpos={float(qpos12[j]):.{precision}f}  action={float(action12[j]):.{precision}f}")
    if float(np.max(np.abs(qpos12))) < 1e-6:
        print("  WARN: |qpos| all near zero; reads likely failed (same as zeros in HDF5 when CAN drops).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Direct local teleop (leader Dynamixel -> follower RobStride via move_actuators), no UDP"
    )
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-baud", type=int, default=57600)
    parser.add_argument("--rate", type=float, default=10.0, help="Control loop rate in Hz")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Read leader and compute targets but do not command follower motors")
    parser.add_argument("--no-print-follower", action="store_true",
                        help="Suppress the simple follower qpos line printed under each leader line "
                             "(uses arm.read_joints12, same as scripts/read_follower_joints.py).")
    parser.add_argument("--print-follower-precision", type=int, default=4,
                        help="Decimals for the simple follower line (default 4).")
    parser.add_argument("--print-follower-qpos", action="store_true",
                        help="On each --print-every loop, read follower qpos via follower_qpos_reader "
                             "(retries + MIT fallback) and print the full diagnostic block.")
    parser.add_argument("--print-follower-qpos-precision", type=int, default=4)
    parser.add_argument("--follower-qpos-after-writes", action="store_true",
                        help="Read encoders after MIT writes (legacy; more CAN contention).")
    parser.add_argument("--follower-can-read-retries", type=int, default=8)
    parser.add_argument("--follower-can-read-retry-delay-ms", type=float, default=1.0)
    parser.add_argument("--follower-mit-sweep-timeout-s", type=float, default=0.4)
    parser.add_argument("--follower-mit-sweep-max-frames", type=int, default=320)
    parser.add_argument("--verbose-follower-can-reads", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ensure_import_paths(project_root)

    from dynamixel_easy_sdk import Connector

    follower_qpos_read_before_writes = not args.follower_qpos_after_writes

    print(f"Opening leader port {args.leader_port} @ {args.leader_baud}...")
    connector = Connector(args.leader_port, args.leader_baud)

    print("Scanning leader motors...")
    leader_motors = connector.createAllMotors()
    if not leader_motors:
        raise RuntimeError("No leader Dynamixel motors found")
    print(f"Found {len(leader_motors)} leader motors: {[m.id for m in leader_motors]}")

    for m in leader_motors:
        try:
            m.disableTorque()
            time.sleep(0.02)
        except Exception as e:
            print(f"  Warning: could not disable leader torque for motor {m.id}: {e}")

    # --- follower arm controller via move_actuators ---
    arm: ActuatorController | None = None
    if not args.dry_run:
        arm = ActuatorController(ramp=True)
        arm.connect()
        # expose buses/motors for ref read and qpos logging below
        left_bus = arm._left_bus
        right_bus = arm._right_bus
        left_motors = arm._left_motors
        right_motors = arm._right_motors
    else:
        left_bus = right_bus = None
        left_motors = [(f"motor_{mid}", mid) for mid in LEFT_ROBSTRIDE_IDS]
        right_motors = [(f"motor_{mid}", mid) for mid in RIGHT_ROBSTRIDE_IDS]

    # --- resilient qpos reader ---
    qpos_reader = None
    if args.print_follower_qpos and not args.dry_run and (left_bus or right_bus):
        from follower_qpos_reader import ResilientFollowerQposReader
        from move_actuators import _load_robstride
        _, _, ParameterType = _load_robstride()
        qpos_reader = ResilientFollowerQposReader(
            ParameterType,
            can_read_retries=args.follower_can_read_retries,
            can_read_retry_delay_s=max(0.0, args.follower_can_read_retry_delay_ms / 1000.0),
            mit_sweep_timeout_s=args.follower_mit_sweep_timeout_s,
            mit_sweep_max_frames=args.follower_mit_sweep_max_frames,
            verbose=args.verbose_follower_can_reads,
        )

    print("Mode:", "DRY-RUN" if args.dry_run else "LIVE")
    print("Left arm:", "enabled" if left_bus else ("skipped" if args.dry_run else "disabled"))
    print("Right arm:", "enabled" if right_bus else ("skipped" if args.dry_run else "disabled"))
    if args.print_follower_qpos and qpos_reader is not None:
        print(
            "Follower qpos: resilient reader; "
            f"read phase={'before writes' if follower_qpos_read_before_writes else 'after writes'} on print ticks."
        )
    print("Align leader and follower, then move leader. Ctrl+C to stop.")

    teleop_initialized = False
    prev_servo = [0.0] * 12
    accum = [0.0] * 12
    robstride_ref: dict[str, float] = {}
    last_ramp_t: float | None = None

    loop_period = 1.0 / max(args.rate, 1e-3)
    loops = 0

    try:
        while True:
            t0 = time.monotonic()
            angles = get_joint_angles_from_motors(leader_motors)
            a12 = pad12(angles)

            if len(angles) < 12:
                time.sleep(loop_period)
                continue

            # --- teleop zero: capture follower refs on first valid sample ---
            if not teleop_initialized:
                if arm is not None and (left_bus or right_bus):
                    refs = arm.read_joints12()
                    for i, (motor_name, _) in enumerate(left_motors):
                        robstride_ref[motor_name] = float(refs[i])
                    for i, (motor_name, _) in enumerate(right_motors):
                        robstride_ref[motor_name] = float(refs[6 + i])
                else:
                    for motor_name, _ in left_motors + right_motors:
                        robstride_ref[motor_name] = 0.0
                # seed arm ramp state from the refs
                if arm is not None:
                    for motor_name, _ in left_motors:
                        arm._ramped[motor_name] = robstride_ref.get(motor_name, 0.0)
                    for motor_name, _ in right_motors:
                        arm._ramped[motor_name] = robstride_ref.get(motor_name, 0.0)
                prev_servo = list(a12)
                teleop_initialized = True
                last_ramp_t = time.monotonic()
                print("Teleop zero set: captured follower refs.")
                continue

            # --- accumulate leader deltas ---
            for i in range(12):
                accum[i] += shortest_delta_units(prev_servo[i], a12[i])
                prev_servo[i] = a12[i]

            now = time.monotonic()
            if last_ramp_t is None:
                last_ramp_t = now
            dt = max(1e-4, min(now - last_ramp_t, ROBSTRIDE_RAMP_DT_MAX_S))
            last_ramp_t = now
            max_step = ROBSTRIDE_RAMP_MAX_SPEED_RAD_S * dt

            # --- compute 12-vector desired targets ---
            targets = np.zeros(12, dtype=np.float64)
            for i, (motor_name, motor_id) in enumerate(left_motors):
                servo_idx = [0, 2, 4, 6, 8, 10][i]
                base = robstride_ref.get(motor_name, 0.0)
                targets[i] = base + accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
            for i, (motor_name, motor_id) in enumerate(right_motors):
                servo_idx = [1, 3, 5, 7, 9, 11][i]
                base = robstride_ref.get(motor_name, 0.0)
                targets[6 + i] = base + accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)

            # --- optional pre-write qpos read ---
            will_print = args.print_every > 0 and (loops + 1) % args.print_every == 0
            qpos_log: np.ndarray | None = None
            if will_print and args.print_follower_qpos and qpos_reader is not None and follower_qpos_read_before_writes:
                qpos_log = qpos_reader.read_qpos12(
                    left_bus, right_bus, left_motors, right_motors, _caller="direct_before_writes"
                )

            # --- send to motors via ActuatorController ---
            if arm is not None:
                arm.command_joints12(targets, ramp=True)
            ramped_cmd = arm._ramped if arm is not None else {}

            # --- optional post-write qpos read ---
            if will_print and args.print_follower_qpos and qpos_reader is not None and not follower_qpos_read_before_writes:
                qpos_log = qpos_reader.read_qpos12(
                    left_bus, right_bus, left_motors, right_motors, _caller="direct_after_writes"
                )

            loops += 1
            if args.print_every > 0 and loops % args.print_every == 0:
                leader_str = "[" + ", ".join(f"{x:.2f}" for x in a12) + "]"
                msg = f"[direct] loop #{loops} leader[:12]={leader_str}"
                if args.dry_run:
                    preview = {k: round(v, 4) for k, v in list(ramped_cmd.items())[:4]}
                    msg += f" | target_preview={preview}"
                print(msg)
                # Simple follower line (matches scripts/read_follower_joints.py behavior).
                if not args.no_print_follower and arm is not None:
                    try:
                        fq = arm.read_joints12()
                        prec = max(0, args.print_follower_precision)
                        follower_str = ", ".join(f"{float(v):.{prec}f}" for v in fq)
                        print(f"             follower[:12]=[{follower_str}]")
                    except Exception as e:
                        print(f"             follower[:12] read failed: {e}")
                if args.print_follower_qpos:
                    action12 = ramped_cmd_to_action12(left_motors, right_motors, ramped_cmd)
                    qpos12 = (
                        np.asarray(qpos_log, dtype=np.float64).reshape(-1)
                        if qpos_log is not None
                        else np.zeros(12, dtype=np.float64)
                    )
                    print_follower_qpos_action_block(
                        loop_n=loops, qpos12=qpos12, action12=action12,
                        precision=max(0, args.print_follower_qpos_precision),
                    )

            elapsed = time.monotonic() - t0
            slp = loop_period - elapsed
            if slp > 0:
                time.sleep(slp)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if qpos_reader is not None:
            print(qpos_reader.stats_line(), flush=True)
        if arm is not None:
            try:
                print(arm.read_stats_line(), flush=True)
            except Exception:
                pass
            arm.disconnect()
            print("RobStride buses disconnected.")
        try:
            connector.closePort()
        except Exception:
            pass


if __name__ == "__main__":
    main()
