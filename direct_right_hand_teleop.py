#!/usr/bin/env python3
"""
Right-hand-only direct teleop (testing).

Leader Dynamixel IDs  2,4,6,8,10,12,14,16  (8 servos)
Follower RobStride IDs 2,4,6,8,10,12,14,16  (right arm 7+1)

1:1 mapping — every follower right joint (including wrist_roll/yaw and gripper)
is driven from its matching leader servo. Left arm is not teleoped: if the left
CAN bus comes up, it is held at the teleop-zero ref (power-on start pose via
``ActuatorController`` turn offsets).

Safety / control path matches ``record/record_episodes_3cam.py`` and
``direct_teleop.py``:
- ``ActuatorController`` with ramp + safety_clamp (default)
- seed ramp from encoders before first command
- optional zero-pose preflight on the right arm
- ``feedback12`` encoder read on every ``command_joints`` tick
- power-on software zero (start-pose offsets) left as-is in ``move_actuators``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from direct_teleop import (
    ARM_DOF,
    GRIPPER_MOTION_SCALE,
    INVERT_DELTA_MOTOR_IDS,
    NUM_JOINTS,
    RIGHT_GRIPPER_MOTOR_ID,
    RIGHT_ROBSTRIDE_IDS,
    ROBSTRIDE_RAMP_DT_MAX_S,
    ROBSTRIDE_RAMP_MAX_SPEED_RAD_S,
    ActuatorController,
    accum_units_to_target_delta_rad,
    ensure_import_paths,
    get_joint_angles_from_motors,
    print_follower_qpos_action_block,
    ramped_cmd_to_action16,
    shortest_delta_units,
    verify_follower_zero_pose,
)

# Ordered leader Dynamixel IDs for the right-hand test rig (1:1 with RIGHT_ROBSTRIDE_IDS).
RIGHT_LEADER_MOTOR_IDS: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16)
RIGHT_LEADER_NUM_JOINTS = len(RIGHT_LEADER_MOTOR_IDS)
assert RIGHT_LEADER_NUM_JOINTS == len(RIGHT_ROBSTRIDE_IDS) == ARM_DOF

# Flip these relative to ``direct_teleop.INVERT_DELTA_MOTOR_IDS`` (record baseline).
# Calibrated for the 7+1 right-hand leader: shoulder_roll (4) and wrist_yaw (14).
RIGHT_HAND_EXTRA_INVERT_MOTOR_IDS: frozenset[int] = frozenset({4, 14})


def right_hand_delta_rad(accum_units: float, motor_id: int) -> float:
    """Leader→follower delta with record invert, then right-hand-only sign flips."""
    delta = accum_units_to_target_delta_rad(float(accum_units), int(motor_id))
    if int(motor_id) in RIGHT_HAND_EXTRA_INVERT_MOTOR_IDS:
        return -delta
    return delta


def select_right_leader_motors(all_motors: list) -> list:
    """Pick Dynamixels with IDs in RIGHT_LEADER_MOTOR_IDS, sorted in that order."""
    by_id = {int(m.id): m for m in all_motors}
    missing = [mid for mid in RIGHT_LEADER_MOTOR_IDS if mid not in by_id]
    if missing:
        found = sorted(by_id.keys())
        raise RuntimeError(
            f"Right-hand leader missing Dynamixel ID(s) {missing}. "
            f"Found IDs: {found}. Expected: {list(RIGHT_LEADER_MOTOR_IDS)}"
        )
    return [by_id[mid] for mid in RIGHT_LEADER_MOTOR_IDS]


def leader8_to_follower16(
    accum8: list[float] | np.ndarray,
    robstride_ref16: dict[str, float] | np.ndarray,
    left_motors: list[tuple[str, int]],
    right_motors: list[tuple[str, int]],
) -> np.ndarray:
    """
    Map right leader 8-DoF accumulated deltas + follower refs → 16-DoF command.

    Right arm: each leader slot i drives right follower slot i (1:1).
    Left arm: held at teleop-zero refs (no leader mapping).
    """
    accum = np.asarray(accum8, dtype=np.float64).reshape(-1)
    if accum.size < RIGHT_LEADER_NUM_JOINTS:
        accum = np.pad(accum, (0, RIGHT_LEADER_NUM_JOINTS - accum.size))

    targets = np.zeros(NUM_JOINTS, dtype=np.float64)

    def _ref(motor_name: str, fallback_idx: int) -> float:
        if isinstance(robstride_ref16, dict):
            return float(robstride_ref16.get(motor_name, 0.0))
        ref = np.asarray(robstride_ref16, dtype=np.float64).reshape(-1)
        return float(ref[fallback_idx]) if fallback_idx < len(ref) else 0.0

    # Left: hold teleop-zero refs (power-on logical 0 if arm was at home at connect).
    for i, (motor_name, _) in enumerate(left_motors):
        targets[i] = _ref(motor_name, i)

    # Right: 1:1 leader → follower (gripper scale via motor 16; extra flips for 4, 14).
    for i, (motor_name, motor_id) in enumerate(right_motors):
        abs_i = ARM_DOF + i
        base = _ref(motor_name, abs_i)
        targets[abs_i] = base + right_hand_delta_rad(float(accum[i]), motor_id)

    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Right-hand-only direct teleop: Dynamixel IDs "
            f"{list(RIGHT_LEADER_MOTOR_IDS)} → RobStride {list(RIGHT_ROBSTRIDE_IDS)}"
        )
    )
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-baud", type=int, default=57600)
    parser.add_argument("--rate", type=float, default=10.0, help="Control loop rate in Hz")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read leader and compute targets but do not command follower motors",
    )
    parser.add_argument(
        "--no-print-follower",
        action="store_true",
        help="Suppress the simple follower qpos line under each leader line.",
    )
    parser.add_argument("--print-follower-precision", type=int, default=4)
    parser.add_argument(
        "--print-follower-qpos",
        action="store_true",
        help="Print full diagnostic qpos/action block via follower_qpos_reader.",
    )
    parser.add_argument("--print-follower-qpos-precision", type=int, default=4)
    parser.add_argument("--follower-qpos-after-writes", action="store_true")
    parser.add_argument("--follower-can-read-retries", type=int, default=8)
    parser.add_argument("--follower-can-read-retry-delay-ms", type=float, default=1.0)
    parser.add_argument("--follower-mit-sweep-timeout-s", type=float, default=0.4)
    parser.add_argument("--follower-mit-sweep-max-frames", type=int, default=320)
    parser.add_argument("--verbose-follower-can-reads", action="store_true")
    parser.add_argument(
        "--skip-zero-pose",
        action="store_true",
        help="Skip right-arm home check before live teleop.",
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

    leader_motors = select_right_leader_motors(all_leader)
    print(
        f"Using right-hand leaders (ordered): {[m.id for m in leader_motors]} → "
        f"followers {list(RIGHT_ROBSTRIDE_IDS)}"
    )
    effective_invert = (
        (INVERT_DELTA_MOTOR_IDS & set(RIGHT_ROBSTRIDE_IDS))
        ^ set(RIGHT_HAND_EXTRA_INVERT_MOTOR_IDS)
    )
    print(
        f"Gripper scale={GRIPPER_MOTION_SCALE} (motor {RIGHT_GRIPPER_MOTOR_ID}); "
        f"effective invert (right) = {sorted(effective_invert)} "
        f"(extra flips vs record: {sorted(RIGHT_HAND_EXTRA_INVERT_MOTOR_IDS)})"
    )

    for m in leader_motors:
        try:
            m.disableTorque()
            time.sleep(0.02)
        except Exception as e:
            print(f"  Warning: could not disable leader torque for motor {m.id}: {e}")

    from move_actuators import LEFT_ROBSTRIDE_IDS

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
        left_bus = arm._left_bus
        right_bus = arm._right_bus
        left_motors = arm._left_motors
        right_motors = arm._right_motors

        if not right_bus or not right_motors:
            arm.disconnect(send_zero=False)
            raise RuntimeError(
                "Right RobStride bus did not come up. Check RIGHT_CAN and motor power."
            )

        encoders = arm.read_joints(samples=3)
        arm.seed_ramp_from_angles(encoders)

        if not args.skip_zero_pose:
            qpos = encoders
            for _ in range(3):
                qpos = arm.read_joints(samples=2)
            # Zero-pose check only on the right arm (indices ARM_DOF .. NUM_JOINTS-1).
            right_q = np.asarray(qpos, dtype=np.float64).reshape(-1)[ARM_DOF:NUM_JOINTS]
            right_ids = [mid for _, mid in right_motors]
            try:
                verify_follower_zero_pose(right_q, right_ids)
                qpos_str = ", ".join(f"{float(v):+.4f}" for v in right_q)
                print(f"[right_teleop] Right zero-pose check OK: qpos=[{qpos_str}]")
            except RuntimeError:
                arm.disconnect(send_zero=False)
                raise
        else:
            print("[right_teleop] Skipping right zero-pose check (--skip-zero-pose).")
    else:
        left_bus = right_bus = None
        left_motors = [(f"motor_{mid}", mid) for mid in LEFT_ROBSTRIDE_IDS]
        right_motors = [(f"motor_{mid}", mid) for mid in RIGHT_ROBSTRIDE_IDS]

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
    print("Left arm:", "hold teleop-zero" if left_bus else ("skipped" if args.dry_run else "disabled"))
    print("Right arm:", "teleop" if right_bus else ("skipped" if args.dry_run else "disabled"))
    print("Align right leader and follower, then move leader. Ctrl+C to stop.")

    teleop_initialized = False
    prev_servo = [0.0] * RIGHT_LEADER_NUM_JOINTS
    accum = [0.0] * RIGHT_LEADER_NUM_JOINTS
    robstride_ref: dict[str, float] = {}

    loop_period = 1.0 / max(args.rate, 1e-3)
    loops = 0

    try:
        while True:
            t0 = time.monotonic()
            angles = get_joint_angles_from_motors(leader_motors)
            if len(angles) < RIGHT_LEADER_NUM_JOINTS:
                time.sleep(loop_period)
                continue
            a8 = [float(x) for x in angles[:RIGHT_LEADER_NUM_JOINTS]]

            if not teleop_initialized:
                if arm is not None and (left_bus or right_bus):
                    refs = arm.read_joints(samples=2)
                    for i, (motor_name, _) in enumerate(left_motors):
                        robstride_ref[motor_name] = float(refs[i])
                    for i, (motor_name, _) in enumerate(right_motors):
                        robstride_ref[motor_name] = float(refs[ARM_DOF + i])
                    arm.seed_ramp_from_angles(refs)
                else:
                    for motor_name, _ in left_motors + right_motors:
                        robstride_ref[motor_name] = 0.0
                prev_servo = list(a8)
                teleop_initialized = True
                print(
                    "Teleop zero set: captured follower refs "
                    "(right = relative deltas; left held at refs)."
                )
                continue

            for i in range(RIGHT_LEADER_NUM_JOINTS):
                accum[i] += shortest_delta_units(prev_servo[i], a8[i])
                prev_servo[i] = a8[i]

            targets = leader8_to_follower16(accum, robstride_ref, left_motors, right_motors)

            will_print = args.print_every > 0 and (loops + 1) % args.print_every == 0
            qpos_log: np.ndarray | None = None
            if (
                will_print
                and args.print_follower_qpos
                and qpos_reader is not None
                and follower_qpos_read_before_writes
            ):
                qpos_log = qpos_reader.read_qpos12(
                    left_bus, right_bus, left_motors, right_motors, _caller="right_before_writes"
                )

            raw_encoder: np.ndarray | None = None
            if arm is not None:
                try:
                    raw_encoder = arm.read_joints(samples=2)
                except Exception as e:
                    print(f"[right_teleop] encoder read failed before command: {e}")
                    raw_encoder = None
                arm.command_joints(targets, ramp=True, feedback12=raw_encoder)
            ramped_cmd = arm._ramped if arm is not None else {}

            if (
                will_print
                and args.print_follower_qpos
                and qpos_reader is not None
                and not follower_qpos_read_before_writes
            ):
                qpos_log = qpos_reader.read_qpos12(
                    left_bus, right_bus, left_motors, right_motors, _caller="right_after_writes"
                )

            loops += 1
            if args.print_every > 0 and loops % args.print_every == 0:
                leader_str = "[" + ", ".join(f"{x:.2f}" for x in a8) + "]"
                msg = f"[right_teleop] loop #{loops} leader8={leader_str}"
                if args.dry_run:
                    preview = {k: round(v, 4) for k, v in list(ramped_cmd.items())[:4]}
                    msg += f" | target_preview={preview}"
                print(msg)
                if not args.no_print_follower and arm is not None:
                    try:
                        fq = raw_encoder if raw_encoder is not None else arm.read_joints()
                        prec = max(0, args.print_follower_precision)
                        # Print right half prominently
                        right_vals = np.asarray(fq, dtype=np.float64).reshape(-1)[ARM_DOF:NUM_JOINTS]
                        follower_str = ", ".join(f"{float(v):.{prec}f}" for v in right_vals)
                        print(f"               follower_right8=[{follower_str}]")
                    except Exception as e:
                        print(f"               follower_right8 read failed: {e}")
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
                        loop_n=loops,
                        qpos16=qpos16[:NUM_JOINTS],
                        action16=action16,
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
