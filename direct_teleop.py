#!/usr/bin/env python3
"""
Direct local teleop: read leader Dynamixels and drive follower RobStride joints.

This merges joint_client.py + joint_server.py into one process on one machine.
No UDP/websocket is used.

Flow:
- Read leader present positions from Dynamixel motors (raw units).
- Use first sample as teleop zero (capture follower mechanical refs).
- Accumulate shortest-path deltas in leader raw units.
- Convert deltas to follower target radians and apply ramp-limited MIT commands.
"""

import argparse
import math
import sys
import time
from pathlib import Path

# -------------------- Teleop constants --------------------

SERVO_UNITS_PER_REV = 4096.0
RAD_PER_SERVO_UNIT = 2.0 * math.pi / SERVO_UNITS_PER_REV

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
MOTOR_KP = {
    1: 180.0, 2: 180.0, 3: 180.0, 4: 180.0, 5: 100.0, 6: 180.0,
    7: 180.0, 8: 180.0, 9: 30.0, 10: 30.0, 11: 30.0, 12: 30.0
}
MOTOR_KD = {
    1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 18.0, 6: 18.0,
    7: 50.0, 8: 50.0, 9: 18.0, 10: 18.0, 11: 30.0, 12: 30.0
}
MOTOR_TORQUE_LIMIT = {
    1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 12.0, 6: 12.0,
    7: 12.0, 8: 12.0, 9: 8.0, 10: 8.0, 11: 8.0, 12: 8.0
}

ROBSTRIDE_RAMP_MAX_SPEED_RAD_S = 6.0
ROBSTRIDE_RAMP_DT_MAX_S = 0.1
GRIPPER_MOTION_SCALE = 5.0
GRIPPER_MOTOR_IDS = {11, 12}
GRIPPER_SERVO_INDICES = {10, 11}


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

    candidate_roots = [project_root, Path.cwd(), Path.cwd().parent]
    for root in candidate_roots:
        robstride_dir = root / "robstride_control"
        if robstride_dir.is_dir() and str(robstride_dir) not in sys.path:
            sys.path.insert(0, str(robstride_dir))


def init_robstride_bus(RobstrideBus, Motor, ParameterType, can_channel, motor_ids):
    if RobstrideBus is None or not motor_ids:
        return None, []

    motor_names = [f"motor_{mid}" for mid in motor_ids]
    motors = {}
    for mid, name in zip(motor_ids, motor_names):
        motors[name] = Motor(id=mid, model=MOTOR_MODEL_MAP.get(mid, "rs-02"))
    calibration = {name: {"direction": 1, "homing_offset": 0.0} for name in motor_names}

    try:
        bus = RobstrideBus(can_channel, motors, calibration)
        bus.connect(handshake=True)
        for motor_name in motor_names:
            bus.enable(motor_name)
            time.sleep(0.1)
        for motor_name, mid in zip(motor_names, motor_ids):
            bus.write(motor_name, ParameterType.POSITION_KP, MOTOR_KP[mid])
            time.sleep(0.05)
            bus.write(motor_name, ParameterType.VELOCITY_KP, MOTOR_KD[mid])
            time.sleep(0.05)
            bus.write(motor_name, ParameterType.TORQUE_LIMIT, MOTOR_TORQUE_LIMIT[mid])
            time.sleep(0.05)
        for motor_name in motor_names:
            bus.write(motor_name, ParameterType.MODE, 0)
            time.sleep(0.05)
        time.sleep(0.2)
        return bus, list(zip(motor_names, motor_ids))
    except Exception as e:
        print("RobStride init failed on {}: {}".format(can_channel, e))
        return None, []


def get_joint_angles_from_motors(motors):
    positions = []
    for m in motors:
        positions.append(float(m.getPresentPosition()))
    return positions


def main():
    parser = argparse.ArgumentParser(description="Direct local teleop (leader Dynamixel -> follower RobStride), no UDP")
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0", help="Leader Dynamixel serial port")
    parser.add_argument("--leader-baud", type=int, default=57600, help="Leader Dynamixel baud")
    parser.add_argument("--rate", type=float, default=10.0, help="Control loop rate in Hz")
    parser.add_argument("--print-every", type=int, default=1, help="Print status every N loops")
    parser.add_argument("--dry-run", action="store_true", help="Read leader and compute targets but do not command follower motors")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ensure_import_paths(project_root)

    from dynamixel_easy_sdk import Connector

    try:
        from robstride_dynamics import RobstrideBus, Motor, ParameterType
    except Exception as e:
        print(f"Warning: failed importing robstride_dynamics package: {e}")
        from robstride_dynamics.bus import RobstrideBus, Motor
        from robstride_dynamics.protocol import ParameterType

    print(f"Opening leader port {args.leader_port} @ {args.leader_baud}...")
    connector = Connector(args.leader_port, args.leader_baud)

    print("Scanning leader motors...")
    leader_motors = connector.createAllMotors()
    if not leader_motors:
        raise RuntimeError("No leader Dynamixel motors found")
    print("Found {} leader motors: {}".format(len(leader_motors), [m.id for m in leader_motors]))

    for m in leader_motors:
        try:
            m.disableTorque()
            time.sleep(0.02)
        except Exception as e:
            print("  Warning: could not disable leader torque for motor {}: {}".format(m.id, e))

    left_bus = right_bus = None
    # Keep motor maps available in dry-run so we can still compute/print target trajectories.
    left_motors = [(f"motor_{mid}", mid) for mid in LEFT_ROBSTRIDE_IDS]
    right_motors = [(f"motor_{mid}", mid) for mid in RIGHT_ROBSTRIDE_IDS]
    if not args.dry_run:
        left_bus, left_motors = init_robstride_bus(RobstrideBus, Motor, ParameterType, LEFT_CAN, LEFT_ROBSTRIDE_IDS)
        right_bus, right_motors = init_robstride_bus(RobstrideBus, Motor, ParameterType, RIGHT_CAN, RIGHT_ROBSTRIDE_IDS)

    print("Mode:", "DRY-RUN (no motor writes)" if args.dry_run else "LIVE")
    print("Left arm:", "enabled" if left_bus else ("skipped" if args.dry_run else "disabled"))
    print("Right arm:", "enabled" if right_bus else ("skipped" if args.dry_run else "disabled"))
    print("Align leader and follower, then move leader. Ctrl+C to stop.")

    teleop_initialized = False
    prev_servo = [0.0] * 12
    accum = [0.0] * 12
    robstride_ref = {}
    ramped_cmd = {}
    last_ramp_t = None

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

            if not teleop_initialized:
                if left_motors:
                    for motor_name, _mid in left_motors:
                        if left_bus:
                            try:
                                robstride_ref[motor_name] = left_bus.read(motor_name, ParameterType.MECHANICAL_POSITION)
                            except Exception as e:
                                print("  left ref read {} failed: {}".format(motor_name, e))
                                robstride_ref[motor_name] = 0.0
                        else:
                            robstride_ref[motor_name] = 0.0
                if right_motors:
                    for motor_name, _mid in right_motors:
                        if right_bus:
                            try:
                                robstride_ref[motor_name] = right_bus.read(motor_name, ParameterType.MECHANICAL_POSITION)
                            except Exception as e:
                                print("  right ref read {} failed: {}".format(motor_name, e))
                                robstride_ref[motor_name] = 0.0
                        else:
                            robstride_ref[motor_name] = 0.0
                prev_servo = list(a12)
                teleop_initialized = True
                ramped_cmd.clear()
                for motor_name, _mid in (left_motors or []) + (right_motors or []):
                    ramped_cmd[motor_name] = float(robstride_ref.get(motor_name, 0.0))
                last_ramp_t = time.monotonic()
                print("Teleop zero set: captured follower refs.")
                continue

            for i in range(12):
                accum[i] += shortest_delta_units(prev_servo[i], a12[i])
                prev_servo[i] = a12[i]

            now = time.monotonic()
            if last_ramp_t is None:
                last_ramp_t = now
            dt = max(1e-4, min(now - last_ramp_t, ROBSTRIDE_RAMP_DT_MAX_S))
            last_ramp_t = now
            max_step = ROBSTRIDE_RAMP_MAX_SPEED_RAD_S * dt

            if left_motors:
                for (motor_name, motor_id), servo_idx in zip(left_motors, [0, 2, 4, 6, 8, 10]):
                    base = robstride_ref.get(motor_name, 0.0)
                    d_rad = accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
                    desired = base + d_rad
                    prev_cmd = ramped_cmd.get(motor_name, desired)
                    target = ramp_toward(prev_cmd, desired, max_step)
                    ramped_cmd[motor_name] = target
                    if left_bus:
                        kp, kd = MOTOR_KP[motor_id], MOTOR_KD[motor_id]
                        try:
                            left_bus.write_operation_frame(motor_name, target, kp, kd, 0.0, 0.0)
                        except Exception as e:
                            print("  left {} failed: {}".format(motor_name, e))

            if right_motors:
                for (motor_name, motor_id), servo_idx in zip(right_motors, [1, 3, 5, 7, 9, 11]):
                    base = robstride_ref.get(motor_name, 0.0)
                    d_rad = accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
                    desired = base + d_rad
                    prev_cmd = ramped_cmd.get(motor_name, desired)
                    target = ramp_toward(prev_cmd, desired, max_step)
                    ramped_cmd[motor_name] = target
                    if right_bus:
                        kp, kd = MOTOR_KP[motor_id], MOTOR_KD[motor_id]
                        try:
                            right_bus.write_operation_frame(motor_name, target, kp, kd, 0.0, 0.0)
                        except Exception as e:
                            print("  right {} failed: {}".format(motor_name, e))

            loops += 1
            if args.print_every > 0 and loops % args.print_every == 0:
                msg = "[direct] loop #{} leader[:12]={}".format(loops, [round(x, 2) for x in a12])
                if args.dry_run and ramped_cmd:
                    preview = {k: round(v, 4) for k, v in list(ramped_cmd.items())[:4]}
                    msg += " | target_preview=" + str(preview)
                print(msg)

            elapsed = time.monotonic() - t0
            sleep_s = loop_period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        for bus, motors in [(left_bus, left_motors), (right_bus, right_motors)]:
            if bus and motors:
                try:
                    for motor_name, _ in motors:
                        bus.write_operation_frame(motor_name, 0.0, 0.0, 0.0, 0.0, 0.0)
                    time.sleep(0.5)
                    for motor_name, _ in motors:
                        bus.disable(motor_name)
                    bus.disconnect()
                    print("RobStride bus disconnected.")
                except Exception as e:
                    print("Cleanup warning:", e)
        try:
            connector.closePort()
        except Exception:
            pass


if __name__ == "__main__":
    main()
