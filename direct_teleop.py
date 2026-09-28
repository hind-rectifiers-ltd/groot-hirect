#!/usr/bin/env python3
"""
Direct local teleop: read leader Dynamixels and drive follower RobStride joints.

This merges joint_client.py + joint_server.py into one process on one machine.
No UDP/websocket is used.

Flow:
- Read leader present positions from Dynamixel motors (raw units).
- Use first sample as teleop zero (capture follower mechanical refs).
- Accumulate shortest-path deltas in leader raw units.
- Map leader 7+1 per arm → follower 7+1 per arm (16D, 1:1 by motor ID) via
  ``ActuatorController`` from ``move_actuators``.

Leader / follower layout:
- Leader: 16 Dynamixels = 7 arm + 1 gripper per side
  (IDs 1,3,5,7,9,11,13,15 left / 2,4,6,8,10,12,14,16 right).
- Follower: 16 RobStride = same IDs, same order (1:1).
- Grippers are motors 15 (left) and 16 (right).

Matches ``record/record_episodes_3cam.py``: relative leader deltas, ramp seed
from encoders, optional zero-pose preflight, and ``feedback12`` on every
``command_joints`` so safety clamp uses a same-tick encoder read.

Optional: ``--print-follower-qpos`` uses ``record/follower_qpos_reader.py``.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Leader (Dynamixel) constants — 7 arm + 1 gripper per side (16 total)
# ---------------------------------------------------------------------------

SERVO_UNITS_PER_REV = 4096.0
RAD_PER_SERVO_UNIT = 2.0 * math.pi / SERVO_UNITS_PER_REV

# Leader Dynamixel → follower gripper gain (same on both sides; right verified).
GRIPPER_MOTION_SCALE = 5.0
LEFT_GRIPPER_MOTION_SCALE = GRIPPER_MOTION_SCALE   # motor 15
RIGHT_GRIPPER_MOTION_SCALE = GRIPPER_MOTION_SCALE  # motor 16

# Ordered leader IDs (1:1 with LEFT/RIGHT_ROBSTRIDE_IDS). Accum vector is L then R.
LEFT_LEADER_MOTOR_IDS: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15)
RIGHT_LEADER_MOTOR_IDS: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16)
LEADER_MOTOR_IDS: tuple[int, ...] = LEFT_LEADER_MOTOR_IDS + RIGHT_LEADER_MOTOR_IDS
LEADER_NUM_JOINTS = len(LEADER_MOTOR_IDS)  # 16

# Indices inside the L-then-R leader accum vector (grippers = last slot per arm).
LEFT_GRIPPER_SERVO_INDEX = len(LEFT_LEADER_MOTOR_IDS) - 1   # 7
RIGHT_GRIPPER_SERVO_INDEX = len(LEFT_LEADER_MOTOR_IDS) + len(RIGHT_LEADER_MOTOR_IDS) - 1  # 15
GRIPPER_SERVO_INDICES = {LEFT_GRIPPER_SERVO_INDEX, RIGHT_GRIPPER_SERVO_INDEX}

# Back-compat names used by older call sites (same as motor-ID order slices).
LEFT_LEADER_SERVO_INDICES = tuple(range(len(LEFT_LEADER_MOTOR_IDS)))
RIGHT_LEADER_SERVO_INDICES = tuple(
    range(len(LEFT_LEADER_MOTOR_IDS), LEADER_NUM_JOINTS)
)

# ---------------------------------------------------------------------------
# Re-export arm layout constants from move_actuators for callers that import
# direct_teleop (e.g. record_episodes_3cam, policy_client_3cam).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from move_actuators import (  # noqa: E402
    ARM_DOF,
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

# Follower gripper CAN IDs (last motor in each 7+1 chain).
LEFT_GRIPPER_MOTOR_ID = 15
RIGHT_GRIPPER_MOTOR_ID = 16
GRIPPER_MOTOR_IDS = {LEFT_GRIPPER_MOTOR_ID, RIGHT_GRIPPER_MOTOR_ID}

# Pre-flight: follower joints must be near home (same window as record_episodes_3cam).
ZERO_POSE_LOW_RAD = -0.2
ZERO_POSE_HIGH_RAD = 0.2

# Leader→follower sign flips (raw Dynamixel delta → follower joint delta).
INVERT_DELTA_MOTOR_IDS = {1, 2, 5, 6, 7, 8, 9, 10, 15, 16}

# Extra flips on top of INVERT_DELTA (7+1 leader calibration).
# Right: shoulder_roll (4), wrist_yaw (14) — verified in direct_right_hand_teleop.
# Left: mirrored joints shoulder_roll (3), wrist_yaw (13).
EXTRA_INVERT_DELTA_MOTOR_IDS: frozenset[int] = frozenset({3, 4, 13, 14})
# Back-compat alias used by direct_right_hand_teleop.
RIGHT_HAND_EXTRA_INVERT_MOTOR_IDS = frozenset({4, 14})

# Constant added to follower command (after all sign flips) for elbow_roll L/R.
FOLLOWER_COMMAND_OFFSET_RAD: dict[int, float] = {7: -0.25, 8: 0.25}

# Legacy names kept so older imports do not break (wrists are driven now).
ZERO_FOLLOWER_MOTOR_IDS: set[int] = set()
ZERO_FOLLOWER_INDICES: tuple[int, ...] = ()
# 1:1 within each arm's 8 slots.
LEADER_TO_FOLLOWER_ARM_INDEX = tuple(range(ARM_DOF))

assert len(LEFT_LEADER_MOTOR_IDS) == len(LEFT_ROBSTRIDE_IDS) == ARM_DOF
assert len(RIGHT_LEADER_MOTOR_IDS) == len(RIGHT_ROBSTRIDE_IDS) == ARM_DOF
assert LEADER_NUM_JOINTS == NUM_JOINTS

# ---------------------------------------------------------------------------
# Helpers kept in this file (leader / teleop-specific logic)
# ---------------------------------------------------------------------------

def select_leader_motors(all_motors: list) -> list:
    """Pick Dynamixels for both arms in L-then-R ID order (1:1 with follower)."""
    by_id = {int(m.id): m for m in all_motors}
    missing = [mid for mid in LEADER_MOTOR_IDS if mid not in by_id]
    if missing:
        found = sorted(by_id.keys())
        raise RuntimeError(
            f"Leader missing Dynamixel ID(s) {missing}. "
            f"Found IDs: {found}. Expected: {list(LEADER_MOTOR_IDS)}"
        )
    return [by_id[mid] for mid in LEADER_MOTOR_IDS]


def pad_leader16(angles):
    """Pad/truncate Dynamixel readings to the 16-DoF leader vector (L then R)."""
    a = [float(x) for x in angles]
    while len(a) < LEADER_NUM_JOINTS:
        a.append(0.0)
    return a[:LEADER_NUM_JOINTS]


# Back-compat aliases used by record_episodes_3cam.py
pad_leader12 = pad_leader16
pad12 = pad_leader16


def shortest_delta_units(prev_u: float, curr_u: float, period: float = SERVO_UNITS_PER_REV) -> float:
    d = float(curr_u) - float(prev_u)
    p = period
    return (d + p / 2.0) % p - p / 2.0


def accum_units_to_target_delta_rad(accum_units: float, motor_id: int, servo_idx: int | None = None) -> float:
    delta_rad = accum_units * RAD_PER_SERVO_UNIT
    if motor_id == LEFT_GRIPPER_MOTOR_ID or servo_idx == LEFT_GRIPPER_SERVO_INDEX:
        delta_rad *= LEFT_GRIPPER_MOTION_SCALE
    elif motor_id == RIGHT_GRIPPER_MOTOR_ID or servo_idx == RIGHT_GRIPPER_SERVO_INDEX:
        delta_rad *= RIGHT_GRIPPER_MOTION_SCALE
    if motor_id in INVERT_DELTA_MOTOR_IDS:
        return -delta_rad
    return delta_rad


def follower_command_offset_rad(motor_id: int) -> float:
    """Extra radians added to the follower target after leader→follower mapping."""
    return float(FOLLOWER_COMMAND_OFFSET_RAD.get(int(motor_id), 0.0))


def leader_delta_rad(accum_units: float, motor_id: int, servo_idx: int | None = None) -> float:
    """
    Full leader→follower delta: base invert, extra 7+1 flips, then elbow offset.
    """
    mid = int(motor_id)
    delta = accum_units_to_target_delta_rad(float(accum_units), mid, servo_idx)
    if mid in EXTRA_INVERT_DELTA_MOTOR_IDS:
        delta = -delta
    return delta + follower_command_offset_rad(mid)


def leader16_to_follower16(
    accum16: list[float] | np.ndarray,
    robstride_ref16: dict[str, float] | np.ndarray,
    left_motors: list[tuple[str, int]],
    right_motors: list[tuple[str, int]],
) -> np.ndarray:
    """
    Map leader 16-DoF accumulated deltas + follower refs → 16-DoF command vector.

    Accum order is left IDs then right IDs (see ``LEADER_MOTOR_IDS``). Each
    follower joint is relative to ``robstride_ref16`` (pose at teleop zero).
    """
    accum = np.asarray(accum16, dtype=np.float64).reshape(-1)
    if accum.size < LEADER_NUM_JOINTS:
        accum = np.pad(accum, (0, LEADER_NUM_JOINTS - accum.size))

    targets = np.zeros(NUM_JOINTS, dtype=np.float64)

    def _ref(motor_name: str, fallback_idx: int) -> float:
        if isinstance(robstride_ref16, dict):
            return float(robstride_ref16.get(motor_name, 0.0))
        ref = np.asarray(robstride_ref16, dtype=np.float64).reshape(-1)
        return float(ref[fallback_idx]) if fallback_idx < len(ref) else 0.0

    for i, (motor_name, motor_id) in enumerate(left_motors):
        base = _ref(motor_name, i)
        targets[i] = base + leader_delta_rad(float(accum[i]), motor_id, i)

    for i, (motor_name, motor_id) in enumerate(right_motors):
        abs_i = ARM_DOF + i
        base = _ref(motor_name, abs_i)
        targets[abs_i] = base + leader_delta_rad(float(accum[abs_i]), motor_id, abs_i)

    return targets


# Back-compat alias
leader12_to_follower16 = leader16_to_follower16


def verify_follower_zero_pose(
    qpos16: np.ndarray | list | tuple,
    motor_ids: list[int],
    *,
    low: float = ZERO_POSE_LOW_RAD,
    high: float = ZERO_POSE_HIGH_RAD,
) -> None:
    """Raise RuntimeError if any follower joint is outside [low, high] rad."""
    q = np.asarray(qpos16, dtype=np.float64).reshape(-1)
    out_of_range: list[tuple[int, float]] = []
    n = min(len(q), len(motor_ids))
    for i in range(n):
        v = float(q[i])
        if not (low <= v <= high):
            out_of_range.append((int(motor_ids[i]), v))
    if out_of_range:
        details = "\n".join(
            f"  motor id {mid} is not zero (qpos={v:+.4f} rad, allowed range [{low}, {high}])"
            for mid, v in out_of_range
        )
        raise RuntimeError(
            "Follower pre-teleop zero-pose check failed:\n"
            + details
            + "\nMove the follower arm(s) back to home position (~0 rad) and re-run, "
            "or pass --skip-zero-pose."
        )


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

    # Leader Dynamixel stack lives in sibling teleop_websocket (or TELEOP_PROJECT_ROOT).
    teleop_roots = []
    env_root = os.environ.get("TELEOP_PROJECT_ROOT", "").strip()
    if env_root:
        teleop_roots.append(Path(env_root))
    teleop_roots.append(project_root.parent / "teleop_websocket")
    for root in teleop_roots:
        teleop_src = root / "src"
        if teleop_src.is_dir() and str(teleop_src) not in sys.path:
            sys.path.insert(0, str(teleop_src))
            break

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

_FOLLOWER_JOINT_LABELS_16 = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",  # wrist_roll  (motor 11)
    "L6",  # wrist_yaw   (motor 13)
    "Lg",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",  # wrist_roll  (motor 12)
    "R6",  # wrist_yaw   (motor 14)
    "Rg",
)


def ramped_cmd_to_action16(left_motors, right_motors, ramped_cmd: dict) -> np.ndarray:
    action = np.zeros(NUM_JOINTS, dtype=np.float64)
    for i, (name, _mid) in enumerate(left_motors):
        action[i] = float(ramped_cmd.get(name, 0.0))
    for i, (name, _mid) in enumerate(right_motors):
        action[ARM_DOF + i] = float(ramped_cmd.get(name, 0.0))
    return action


# Back-compat name
ramped_cmd_to_action12 = ramped_cmd_to_action16


def _format_joint_vector_line(name: str, row: np.ndarray, *, precision: int) -> str:
    r = np.asarray(row, dtype=np.float64).reshape(-1)
    inner = ", ".join(f"{float(v):.{precision}f}" for v in r)
    return f"{name} = np.array([{inner}])  # len={len(r)}"


def print_follower_qpos_action_block(*, loop_n: int, qpos16: np.ndarray, action16: np.ndarray, precision: int) -> None:
    print(f"\n--- [direct_teleop] loop={loop_n} ---")
    print(_format_joint_vector_line("qpos", qpos16, precision=precision))
    print(_format_joint_vector_line("action", action16, precision=precision))
    print("per_joint (index label qpos action):")
    n = min(NUM_JOINTS, len(qpos16), len(action16), len(_FOLLOWER_JOINT_LABELS_16))
    for j in range(n):
        lab = _FOLLOWER_JOINT_LABELS_16[j]
        print(f"  [{j:2d}] {lab:4s}  qpos={float(qpos16[j]):.{precision}f}  action={float(action16[j]):.{precision}f}")
    if float(np.max(np.abs(qpos16))) < 1e-6:
        print("  WARN: |qpos| all near zero; reads likely failed (same as zeros in HDF5 when CAN drops).")


# Back-compat name used by older call sites
print_follower_qpos_action_block_12 = print_follower_qpos_action_block


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
                             "(uses arm.read_joints, same as scripts/read_follower_joints.py).")
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
    parser.add_argument(
        "--skip-zero-pose",
        action="store_true",
        help="Skip follower home check before live teleop (record always enforces this).",
    )
    parser.add_argument(
        "--safety-abort",
        action="store_true",
        help="Abort and disconnect on safety breach instead of clamping (default: clamp like record).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ensure_import_paths(project_root)

    from dynamixel_easy_sdk import Connector

    follower_qpos_read_before_writes = not args.follower_qpos_after_writes

    print(f"Opening leader port {args.leader_port} @ {args.leader_baud}...")
    connector = Connector(args.leader_port, args.leader_baud)

    print("Scanning leader motors...")
    all_leader = connector.createAllMotors()
    if not all_leader:
        raise RuntimeError("No leader Dynamixel motors found")
    print(f"Found {len(all_leader)} leader motors: {[m.id for m in all_leader]}")
    leader_motors = select_leader_motors(all_leader)
    print(
        f"Using leaders (L then R): {[m.id for m in leader_motors]} → "
        f"followers {list(LEFT_ROBSTRIDE_IDS) + list(RIGHT_ROBSTRIDE_IDS)} (1:1). "
        f"extra_invert={sorted(EXTRA_INVERT_DELTA_MOTOR_IDS)}; "
        f"offsets={FOLLOWER_COMMAND_OFFSET_RAD}"
    )

    for m in leader_motors:
        try:
            m.disableTorque()
            time.sleep(0.02)
        except Exception as e:
            print(f"  Warning: could not disable leader torque for motor {m.id}: {e}")

    # --- follower arm controller via move_actuators ---
    arm: ActuatorController | None = None
    if not args.dry_run:
        arm = ActuatorController(
            ramp=True,
            ramp_max_speed_rad_s=ROBSTRIDE_RAMP_MAX_SPEED_RAD_S,
            ramp_dt_max_s=ROBSTRIDE_RAMP_DT_MAX_S,
            safety_clamp=not args.safety_abort,
            safety_abort_on_breach=bool(args.safety_abort),
            read_max_retries=4,
            parallel_bus_reads=True,
        )
        arm.connect()
        # expose buses/motors for ref read and qpos logging below
        left_bus = arm._left_bus
        right_bus = arm._right_bus
        left_motors = arm._left_motors
        right_motors = arm._right_motors
        # Same as record: seed ramp from encoders so first command does not snap.
        encoders = arm.read_joints(samples=3)
        arm.seed_ramp_from_angles(encoders)
        if not args.skip_zero_pose:
            qpos = encoders
            for _ in range(3):
                qpos = arm.read_joints(samples=2)
            motor_ids = [mid for _, mid in left_motors] + [mid for _, mid in right_motors]
            try:
                verify_follower_zero_pose(qpos, motor_ids)
                qpos_str = ", ".join(f"{float(v):+.4f}" for v in qpos)
                print(f"[direct] Follower zero-pose check OK: qpos=[{qpos_str}]")
            except RuntimeError:
                arm.disconnect(send_zero=False)
                raise
        else:
            print("[direct] Skipping follower zero-pose check (--skip-zero-pose).")
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
    prev_servo = [0.0] * LEADER_NUM_JOINTS
    accum = [0.0] * LEADER_NUM_JOINTS
    robstride_ref: dict[str, float] = {}
    last_ramp_t: float | None = None

    loop_period = 1.0 / max(args.rate, 1e-3)
    loops = 0

    try:
        while True:
            t0 = time.monotonic()
            angles = get_joint_angles_from_motors(leader_motors)
            a16 = pad_leader16(angles)

            if len(angles) < LEADER_NUM_JOINTS:
                time.sleep(loop_period)
                continue

            # --- teleop zero: capture follower refs on first valid sample ---
            if not teleop_initialized:
                if arm is not None and (left_bus or right_bus):
                    refs = arm.read_joints(samples=2)
                    for i, (motor_name, _) in enumerate(left_motors):
                        robstride_ref[motor_name] = float(refs[i])
                    for i, (motor_name, _) in enumerate(right_motors):
                        robstride_ref[motor_name] = float(refs[ARM_DOF + i])
                    # Keep control references in the motor's native encoder frame
                    # (same as record_episodes_3cam).
                    arm.seed_ramp_from_angles(refs)
                else:
                    for motor_name, _ in left_motors + right_motors:
                        robstride_ref[motor_name] = 0.0
                prev_servo = list(a16)
                teleop_initialized = True
                last_ramp_t = time.monotonic()
                print("Teleop zero set: captured follower refs (relative deltas from here).")
                continue

            # --- accumulate leader deltas (raw Dynamixel units, shortest path) ---
            for i in range(LEADER_NUM_JOINTS):
                accum[i] += shortest_delta_units(prev_servo[i], a16[i])
                prev_servo[i] = a16[i]

            now = time.monotonic()
            if last_ramp_t is None:
                last_ramp_t = now
            last_ramp_t = now

            # --- compute 16-vector desired targets (1:1 leader → follower) ---
            targets = leader16_to_follower16(accum, robstride_ref, left_motors, right_motors)

            # --- optional diagnostic qpos (before writes when requested) ---
            will_print = args.print_every > 0 and (loops + 1) % args.print_every == 0
            qpos_log: np.ndarray | None = None
            if will_print and args.print_follower_qpos and qpos_reader is not None and follower_qpos_read_before_writes:
                qpos_log = qpos_reader.read_qpos12(
                    left_bus, right_bus, left_motors, right_motors, _caller="direct_before_writes"
                )

            # --- send to motors: same-tick encoder feedback for safety (like record) ---
            raw_encoder: np.ndarray | None = None
            if arm is not None:
                try:
                    raw_encoder = arm.read_joints(samples=2)
                except Exception as e:
                    print(f"[direct] encoder read failed before command: {e}")
                    raw_encoder = None
                arm.command_joints(targets, ramp=True, feedback12=raw_encoder)
            ramped_cmd = arm._ramped if arm is not None else {}

            # --- optional post-write qpos read ---
            if will_print and args.print_follower_qpos and qpos_reader is not None and not follower_qpos_read_before_writes:
                qpos_log = qpos_reader.read_qpos12(
                    left_bus, right_bus, left_motors, right_motors, _caller="direct_after_writes"
                )

            loops += 1
            if args.print_every > 0 and loops % args.print_every == 0:
                leader_str = "[" + ", ".join(f"{x:.2f}" for x in a16) + "]"
                msg = f"[direct] loop #{loops} leader16={leader_str}"
                if args.dry_run:
                    preview = {k: round(v, 4) for k, v in list(ramped_cmd.items())[:4]}
                    msg += f" | target_preview={preview}"
                print(msg)
                # Simple follower line — reuse same-tick read when available.
                if not args.no_print_follower and arm is not None:
                    try:
                        fq = raw_encoder if raw_encoder is not None else arm.read_joints()
                        prec = max(0, args.print_follower_precision)
                        follower_str = ", ".join(f"{float(v):.{prec}f}" for v in fq)
                        print(f"             follower[:{NUM_JOINTS}]=[{follower_str}]")
                    except Exception as e:
                        print(f"             follower[:{NUM_JOINTS}] read failed: {e}")
                if args.print_follower_qpos:
                    action16 = ramped_cmd_to_action16(left_motors, right_motors, ramped_cmd)
                    qpos16 = (
                        np.asarray(qpos_log, dtype=np.float64).reshape(-1)
                        if qpos_log is not None
                        else (
                            np.asarray(raw_encoder, dtype=np.float64).reshape(-1)
                            if raw_encoder is not None
                            else np.zeros(NUM_JOINTS, dtype=np.float64)
                        )
                    )
                    if qpos16.size < NUM_JOINTS:
                        qpos16 = np.pad(qpos16, (0, NUM_JOINTS - qpos16.size))
                    print_follower_qpos_action_block(
                        loop_n=loops, qpos16=qpos16[:NUM_JOINTS], action16=action16,
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
