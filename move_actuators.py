"""
Independent module: drive 16 RobStride follower arm joints from a position vector.

This file has no dependency on any other file in this repo.  It can be imported
from any script.

Each arm is 7 revolute joints + 1 gripper (7 + 1 DoF), matching the humanoid
URDF.  Per-arm joint order:

    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_roll,
    wrist_pitch, wrist_roll, wrist_yaw, gripper

Joint vector order (radians):
  [0..7]   left arm  — Waveshare ``zcan1`` (hardware CAN2), motor IDs 1, 3, 5, 7, 9, 11, 13, 15
  [8..15]  right arm — Waveshare ``zcan0`` (hardware CAN1), motor IDs 2, 4, 6, 8, 10, 12, 14, 16

CAN channels: Waveshare USB-CAN-FD-B via ``zcan0`` / ``zcan1`` (no ``ip link``).
Override with env ``ROBSTRIDE_LEFT_CAN`` / ``ROBSTRIDE_RIGHT_CAN`` (e.g. SocketCAN ``can1`` / ``can0``).

The two wrist motors added per arm (wrist_roll / wrist_yaw) reuse the rs-02
model and the same tuning (kp / kd / torque limit) as motors 9 and 10. The
gripper is the last motor in each chain (ID 15 left, ID 16 right).

Typical use::

    from move_actuators import ActuatorController
    import numpy as np

    targets = np.zeros(16)
    targets[2] = 0.5   # left shoulder_yaw
    targets[12] = -0.3 # right shoulder_yaw

    with ActuatorController() as arm:
        # inside a control loop:
        sent = arm.command_joints(targets)

One-shot helper (connects → sends → disconnects)::

    from move_actuators import command_joints
    command_joints([0.0] * 16)

CLI::

    uv run python move_actuators.py --joints 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants (same values as direct_teleop.py)
# ---------------------------------------------------------------------------

# Per-arm joint order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_roll,
#                      wrist_pitch, wrist_roll, wrist_yaw, gripper
# IDs 11/13 (left) and 12/14 (right) are the new wrist_roll / wrist_yaw motors.
# Grippers are the last motor in each chain: ID 15 (left), ID 16 (right).
# If you flashed motors with different CAN IDs, edit the two lists below.
LEFT_ROBSTRIDE_IDS: list[int] = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT_ROBSTRIDE_IDS: list[int] = [2, 4, 6, 8, 10, 12, 14, 16]
# Waveshare USB-CAN-FD-B: hardware CAN2 -> zcan1 (left), CAN1 -> zcan0 (right).
# Legacy gs_usb SocketCAN: can1 (left), can0 (right) — set via env vars below.
LEFT_CAN = os.environ.get("ROBSTRIDE_LEFT_CAN", "zcan1")
RIGHT_CAN = os.environ.get("ROBSTRIDE_RIGHT_CAN", "zcan0")

MOTOR_MODEL_MAP: dict[int, str] = {
    1: "rs-03", 2: "rs-03", 3: "rs-03", 4: "rs-03",
    5: "rs-06", 6: "rs-06", 7: "rs-06", 8: "rs-06",
    9: "rs-02", 10: "rs-02",                      # wrist_pitch
    11: "rs-02", 12: "rs-02", 13: "rs-02", 14: "rs-02",  # wrist_roll / wrist_yaw
    15: "rs-02", 16: "rs-02",                     # grippers
}
MOTOR_KP: dict[int, float] = {
    1: 180.0, 2: 180.0, 3: 180.0, 4: 180.0, 5: 100.0, 6: 100.0,
    7: 180.0, 8: 180.0, 9: 30.0,  10: 30.0,
    11: 30.0, 12: 30.0, 13: 30.0, 14: 30.0,       # wrist_roll / wrist_yaw (as 9/10)
    15: 30.0, 16: 30.0,                            # grippers
}
MOTOR_KD: dict[int, float] = {
    1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 18.0, 6: 18.0,
    7: 50.0, 8: 50.0, 9: 18.0, 10: 18.0,
    11: 18.0, 12: 18.0, 13: 18.0, 14: 18.0,       # wrist_roll / wrist_yaw (as 9/10)
    15: 30.0, 16: 30.0,                            # grippers
}
MOTOR_TORQUE_LIMIT: dict[int, float] = {
    1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 12.0, 6: 12.0,
    7: 12.0, 8: 12.0, 9: 8.0,  10: 8.0,
    11: 8.0, 12: 8.0, 13: 8.0, 14: 8.0,
    15: 8.0, 16: 8.0,                              # grippers
}

# Per-motor rotation sign, applied symmetrically to both commands (write) and
# encoder reads. Use -1.0 when a motor is mounted opposite the URDF / IK joint
# convention so a positive joint angle drives the joint the correct way.
#
# This is applied inside move_actuators (not the driver calibration) because the
# encoder is read via bus.read(MECHANICAL_POSITION), which does NOT apply the
# driver's direction calibration; doing it here keeps write and read in the same
# logical frame (so the safety delta check stays correct).
#
# Motor 8 = right arm elbow_roll (RIGHT_ROBSTRIDE_IDS index 3).
MOTOR_DIRECTION: dict[int, float] = {
    1: 1.0, 2: 1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: -1.0,
    8: 1.0,                                        # right elbow_roll (inverted mounting)
    9: 1.0, 10: 1.0, 11: -1.0, 12: -1.0, 13: 1.0, 14: 1.0,
    15: -1.0, 16: 1.0,
}

# Per-motor "software zero": the joint angle (rad, logical frame) the motor sits
# at when it reads its mechanical zero. RobStride absolute encoders have a known
# glitch where a motor truly at angle θ intermittently reports θ ± 2π (e.g. a
# joint at 0 reads ~6.28). Every revolute joint here travels < π from its zero
# (largest range is the elbow at -2.30 rad), so any reading more than π away
# from the software zero can only be that wrap glitch. ``_normalize_near_zero``
# folds such readings back into the ±π window around the zero, removing the
# glitch unambiguously. Set a non-zero value here only if a joint's mechanical
# zero is offset from its logical zero.
MOTOR_SOFTWARE_ZERO: dict[int, float] = {mid: 0.0 for mid in range(1, 17)}

RAMP_MAX_SPEED_RAD_S = 6.0   # rad/s slew limit (per joint, per second)
RAMP_DT_MAX_S = 0.1          # cap on dt used for ramp step calculation

# Command safety: compare targets to live encoder reads in the motor's native frame.
# Uses per-step delta limits (not abs(angle) > pi) so wrapped encoders near 2*pi do not false-trip.
SAFETY_MAX_DELTA_RAD = 1.0           # max |ramped target - encoder| per command tick
SAFETY_MAX_INITIAL_DELTA_RAD = 1.0   # stricter limit on the first command after connect
SAFETY_EXCLUDED_MOTOR_IDS: tuple[int, ...] = (15, 16)  # grippers

ARM_DOF = len(LEFT_ROBSTRIDE_IDS)              # joints per arm (7 revolute + gripper = 8)
NUM_JOINTS = len(LEFT_ROBSTRIDE_IDS) + len(RIGHT_ROBSTRIDE_IDS)
JOINT_MOTOR_IDS: list[int] = LEFT_ROBSTRIDE_IDS + RIGHT_ROBSTRIDE_IDS


class SafetyLimitBreachError(RuntimeError):
    """Command rejected because a joint target jumped too far from the current encoder reading."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_robstride_on_path() -> None:
    """Add robstride_control to sys.path so robstride_dynamics can be imported."""
    repo = Path(__file__).resolve().parent
    for root in (repo, Path.cwd(), Path.cwd().parent):
        rd = root / "robstride_control"
        if rd.is_dir() and str(rd) not in sys.path:
            sys.path.insert(0, str(rd))
    src = repo / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_robstride():
    """Import and return (RobstrideBus, Motor, ParameterType)."""
    _ensure_robstride_on_path()
    try:
        from robstride_dynamics import Motor, ParameterType, RobstrideBus
    except Exception:
        from robstride_dynamics.bus import Motor, RobstrideBus
        from robstride_dynamics.protocol import ParameterType
    return RobstrideBus, Motor, ParameterType


def _open_bus(RobstrideBus, Motor, ParameterType, can_channel: str, motor_ids: list[int]):
    """Connect one CAN bus, enable all motors, write gains.  Returns (bus, motor_list)."""
    if not motor_ids:
        return None, []

    motor_names = [f"motor_{mid}" for mid in motor_ids]
    motors_cfg = {
        name: Motor(id=mid, model=MOTOR_MODEL_MAP.get(mid, "rs-02"))
        for mid, name in zip(motor_ids, motor_names)
    }
    calibration = {name: {"direction": 1, "homing_offset": 0.0} for name in motor_names}

    try:
        bus = RobstrideBus(can_channel, motors_cfg, calibration)
        bus.connect(handshake=True)
        for name in motor_names:
            bus.enable(name)
            time.sleep(0.1)
        for name, mid in zip(motor_names, motor_ids):
            bus.write(name, ParameterType.POSITION_KP, MOTOR_KP[mid])
            time.sleep(0.05)
            bus.write(name, ParameterType.VELOCITY_KP, MOTOR_KD[mid])
            time.sleep(0.05)
            bus.write(name, ParameterType.TORQUE_LIMIT, MOTOR_TORQUE_LIMIT[mid])
            time.sleep(0.05)
        for name in motor_names:
            bus.write(name, ParameterType.MODE, 0)
            time.sleep(0.05)
        time.sleep(0.2)
        return bus, list(zip(motor_names, motor_ids))
    except Exception as exc:
        print(f"[move_actuators] bus init failed on {can_channel}: {exc}")
        return None, []


def _shortest_delta_rad(to_angle: float, from_angle: float) -> float:
    """Signed shortest rotation from ``from_angle`` to ``to_angle`` (rad)."""
    d = float(to_angle) - float(from_angle)
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _normalize_near_zero(value: float, zero: float = 0.0) -> float:
    """Collapse RobStride ~2π encoder-wrap glitches around a software zero.

    Maps ``value`` to the equivalent angle within ±π of ``zero``, so a motor
    physically at ``zero`` that spuriously reports ``zero ± 2π`` reads back as
    ``zero``. Safe because every revolute joint here travels less than π from
    its zero, so the ±π window contains exactly one valid representative.
    """
    return float(zero) + _shortest_delta_rad(value, zero)


def _ramp_toward(current: float, desired: float, max_step: float) -> float:
    """Step toward ``desired`` along the shortest angular path (handles ~2π encoder wraps)."""
    err = _shortest_delta_rad(desired, current)
    if abs(err) <= max_step:
        return current + err
    return current + (max_step if err > 0.0 else -max_step)


def _pad12(q: np.ndarray | list | tuple) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.size > NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joint angles, got {arr.size}")
    if arr.size < NUM_JOINTS:
        arr = np.pad(arr, (0, NUM_JOINTS - arr.size))
    return arr.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ActuatorController:
    """
    Persistent connection to the two RobStride CAN buses.

    Open once, call ``command_joints`` as fast as your loop runs,
    close when done.  Safe to use as a context manager::

        with ActuatorController() as arm:
            for targets in trajectory:
                arm.command_joints(targets)
    """

    def __init__(
        self,
        *,
        ramp: bool = True,
        ramp_max_speed_rad_s: float = RAMP_MAX_SPEED_RAD_S,
        ramp_dt_max_s: float = RAMP_DT_MAX_S,
        safety_enabled: bool = True,
        safety_max_delta_rad: float = SAFETY_MAX_DELTA_RAD,
        safety_max_initial_delta_rad: float = SAFETY_MAX_INITIAL_DELTA_RAD,
        safety_excluded_motor_ids: tuple[int, ...] = SAFETY_EXCLUDED_MOTOR_IDS,
        safety_abort_on_breach: bool = True,
        read_max_retries: int = 4,
        parallel_bus_reads: bool = True,
    ):
        """
        Args:
            ramp: Slew-limit each joint toward the target (recommended; prevents jerks).
            ramp_max_speed_rad_s: Maximum joint speed allowed by the ramp (rad/s).
            ramp_dt_max_s: dt is capped at this value when computing ramp step.
            safety_enabled: Reject commands whose targets jump too far from encoder feedback.
            safety_max_delta_rad: Per-tick |target - encoder| limit (rad, native encoder frame).
            safety_max_initial_delta_rad: Limit for the first command after :meth:`connect`.
            safety_excluded_motor_ids: Motor IDs skipped by safety delta checks.
            safety_abort_on_breach: If True, disable torque and disconnect on breach.
            read_max_retries: Per-motor MECHANICAL_POSITION retries after stale RX frames.
            parallel_bus_reads: Read left and right CAN halves in parallel when both are live.
        """
        self._ramp = bool(ramp)
        self._ramp_max_speed = float(ramp_max_speed_rad_s)
        self._ramp_dt_max = float(ramp_dt_max_s)
        self._safety_enabled = bool(safety_enabled)
        self._safety_max_delta_rad = float(max(0.0, safety_max_delta_rad))
        self._safety_max_initial_delta_rad = float(max(0.0, safety_max_initial_delta_rad))
        self._safety_excluded_motor_ids = {int(mid) for mid in safety_excluded_motor_ids}
        self._safety_abort_on_breach = bool(safety_abort_on_breach)
        self._safety_command_count = 0
        self._read_max_retries = max(0, int(read_max_retries))
        self._parallel_bus_reads = bool(parallel_bus_reads)
        self._read_stats_lock = threading.Lock()

        self._left_bus = None
        self._right_bus = None
        self._left_motors: list[tuple[str, int]] = []
        self._right_motors: list[tuple[str, int]] = []
        self._ramped: dict[str, float] = {}
        self._last_cmd_t: float | None = None
        self._connected = False

        # Read stats — useful when MIT writes and register reads compete for the bus.
        self._read_calls = 0       # number of times read_joints() was invoked
        self._read_attempts = 0    # per-joint attempts (16 per call when both buses are live)
        self._read_failures = 0    # per-joint failures (left at 0.0)
        self._rx_frames_drained = 0  # cumulative stale frames flushed before reads

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open both CAN buses and enable all motors."""
        if self._connected:
            return
        RobstrideBus, Motor, ParameterType = _load_robstride()
        self._ParameterType = ParameterType

        self._left_bus, self._left_motors = _open_bus(
            RobstrideBus, Motor, ParameterType, LEFT_CAN, LEFT_ROBSTRIDE_IDS
        )
        self._right_bus, self._right_motors = _open_bus(
            RobstrideBus, Motor, ParameterType, RIGHT_CAN, RIGHT_ROBSTRIDE_IDS
        )
        if not self._left_bus and not self._right_bus:
            raise RuntimeError(
                "[move_actuators] No RobStride bus connected. Check CAN interfaces and motor power."
            )
        self._ramped.clear()
        self._last_cmd_t = time.monotonic()
        self._safety_command_count = 0
        self._connected = True

    def disconnect(self, *, send_zero: bool = True) -> None:
        """
        Release the buses.

        Args:
            send_zero: Send a zero-torque MIT frame before disabling (recommended).
        """
        if not self._connected:
            return
        for bus, motors in ((self._left_bus, self._left_motors), (self._right_bus, self._right_motors)):
            if bus is None or not motors:
                continue
            try:
                if send_zero:
                    for name, _ in motors:
                        bus.write_operation_frame(name, 0.0, 0.0, 0.0, 0.0, 0.0)
                    time.sleep(0.3)
                for name, _ in motors:
                    bus.disable(name)
                bus.disconnect()
            except Exception as exc:
                print(f"[move_actuators] disconnect warning: {exc}")
        self._left_bus = self._right_bus = None
        self._left_motors = []
        self._right_motors = []
        self._ramped.clear()
        self._safety_command_count = 0
        self._connected = False

    def _joint_bus_live(self, joint_index: int) -> bool:
        if joint_index < ARM_DOF:
            return self._left_bus is not None and len(self._left_motors) > 0
        return self._right_bus is not None and len(self._right_motors) > 0

    def _enforce_command_safety(
        self,
        desired12: np.ndarray,
        *,
        feedback12: np.ndarray | list | tuple | None = None,
    ) -> None:
        """
        Abort if any ramp-limited target is too far from the live encoder.

        Uses shortest angular distance so joints near ±2π (e.g. encoder 6.25 rad vs
        0.02 rad) are treated as ~0.05 rad apart, not ~6.2 rad.

        Args:
            feedback12: Optional encoder-frame joint vector from a read in the same
                        control tick. When provided, skips a second ``read_joints12``
                        call (saves ~10 ms per command at 30 Hz).
        """
        if not self._safety_enabled:
            self._safety_command_count += 1
            return

        limit = (
            self._safety_max_initial_delta_rad
            if self._safety_command_count == 0
            else self._safety_max_delta_rad
        )
        self._safety_command_count += 1

        if limit <= 0.0:
            return

        if feedback12 is not None:
            current = _pad12(feedback12)
        else:
            current = self.read_joints()
        breaches: list[tuple[int, float, float, float]] = []
        for i in range(NUM_JOINTS):
            if not self._joint_bus_live(i):
                continue
            mid = JOINT_MOTOR_IDS[i]
            if mid in self._safety_excluded_motor_ids:
                continue
            enc = float(current[i])
            tgt = float(desired12[i])
            delta = abs(_shortest_delta_rad(tgt, enc))
            if delta > limit:
                breaches.append((mid, enc, tgt, delta))

        if not breaches:
            return

        which = "first command after connect" if self._safety_command_count == 1 else "command tick"
        details = "\n".join(
            f"  motor id {mid}: encoder={enc:+.4f} rad  target={tgt:+.4f} rad  "
            f"|delta|={delta:+.4f} rad  (limit {limit:.4f})"
            for mid, enc, tgt, delta in breaches
        )
        msg = (
            "Safety limits breached: joint target jump too large in encoder frame "
            f"({which}).\n"
            + details
            + "\nRefusing to command motors. Check frame/unwrapping and homing."
        )
        if self._safety_abort_on_breach:
            try:
                self.disconnect(send_zero=True)
            except Exception as exc:
                print(f"[move_actuators] safety disconnect warning: {exc}", flush=True)
        raise SafetyLimitBreachError(msg)

    @property
    def connected(self) -> bool:
        return self._connected

    def __enter__(self) -> ActuatorController:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Commanding
    # ------------------------------------------------------------------

    def command_joints(
        self,
        angles12: np.ndarray | list | tuple,
        *,
        ramp: bool | None = None,
        feedback12: np.ndarray | list | tuple | None = None,
    ) -> np.ndarray:
        """
        Send MIT position targets to all 16 joints.

        Args:
            angles12: Target joint angles in radians.  Length must be 16.
                      Order: [L0..L6, Lg, R0..R6, Rg] where 0..6 are
                      shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_roll,
                      wrist_pitch, wrist_roll, wrist_yaw.
            ramp: Override the instance-level ramp setting for this call only.
            feedback12: Optional encoder-frame qpos from the same tick (see
                        :meth:`_enforce_command_safety`).

        Returns:
            The 16 targets actually written to the motors (after ramp limiting).

        Raises:
            RuntimeError: if not connected.
            SafetyLimitBreachError: if a ramp-limited target is too far from the encoder reading.
        """
        if not self._connected:
            raise RuntimeError(
                "[move_actuators] Not connected. Call connect() or use a with-block first."
            )

        q = _pad12(angles12)
        use_ramp = self._ramp if ramp is None else bool(ramp)

        # Compute ramp step from elapsed time
        now = time.monotonic()
        if self._last_cmd_t is None:
            self._last_cmd_t = now
        ramp_dt = max(1e-4, min(now - self._last_cmd_t, self._ramp_dt_max))
        self._last_cmd_t = now
        max_step = self._ramp_max_speed * ramp_dt if use_ramp else float("inf")

        sent = np.zeros(NUM_JOINTS, dtype=np.float64)

        for i, (name, _mid) in enumerate(self._left_motors):
            desired = float(q[i])
            target = _ramp_toward(self._ramped.get(name, desired), desired, max_step)
            self._ramped[name] = target
            sent[i] = target

        for i, (name, _mid) in enumerate(self._right_motors):
            desired = float(q[ARM_DOF + i])
            target = _ramp_toward(self._ramped.get(name, desired), desired, max_step)
            self._ramped[name] = target
            sent[ARM_DOF + i] = target

        # Safety applies to ramp-limited targets actually sent, not the full policy horizon.
        self._enforce_command_safety(sent, feedback12=feedback12)

        for i, (name, mid) in enumerate(self._left_motors):
            if self._left_bus:
                try:
                    self._left_bus.write_operation_frame(
                        name, sent[i] * MOTOR_DIRECTION.get(mid, 1.0),
                        MOTOR_KP[mid], MOTOR_KD[mid], 0.0, 0.0
                    )
                except Exception:
                    pass

        for i, (name, mid) in enumerate(self._right_motors):
            if self._right_bus:
                try:
                    self._right_bus.write_operation_frame(
                        name, sent[ARM_DOF + i] * MOTOR_DIRECTION.get(mid, 1.0),
                        MOTOR_KP[mid], MOTOR_KD[mid], 0.0, 0.0
                    )
                except Exception:
                    pass

        return sent

    # Backward-compatible alias (this controller now spans 16 joints, not 12).
    command_joints12 = command_joints

    def seed_ramp_from_angles(self, angles12: np.ndarray | list | tuple) -> None:
        """
        Pre-load the internal ramp state with the given joint angles.

        After calling this, the next ``command_joints`` will ramp *from* these
        angles toward the requested target instead of snapping straight to it.
        Typical use: read current encoder positions and call this before the
        first command of an inference / playback loop so the arm does not jerk
        from a stale (or zero) ramp seed.

        Args:
            angles12: 16-vector of seed angles, ordered like ``command_joints``.
        """
        q = _pad12(angles12)
        for i, (name, _mid) in enumerate(self._left_motors):
            self._ramped[name] = float(q[i])
        for i, (name, _mid) in enumerate(self._right_motors):
            self._ramped[name] = float(q[ARM_DOF + i])
        self._last_cmd_t = None

    # ------------------------------------------------------------------
    # Convenience: read current mechanical positions
    # ------------------------------------------------------------------

    def _drain_bus_rx(
        self,
        bus,
        *,
        max_frames: int = 256,
        settle_timeout_s: float = 0.002,
        settle_passes: int = 3,
    ) -> int:
        """
        Flush leftover frames from the python-can RX queue (typically stale OPERATION_STATUS
        frames produced by recent write_operation_frame calls).  Returns count drained.

        Without this, ``bus.read(MECHANICAL_POSITION)`` consumes the queue head, which is a
        stale status frame, raises AssertionError, and the register-read response is lost.
        After ~70 cycles the queue is permanently full of stale frames and every read fails.

        We bypass ``RobstrideBus.receive()`` and talk to the underlying ``python-can`` Bus
        directly because the vendor ``receive(timeout=0.0)`` is buggy when the queue is
        empty (its while-loop never executes and ``frame`` is referenced unbound).

        Strategy:
          1. Greedy non-blocking drain to flush whatever is already queued.
          2. Up to ``settle_passes`` short blocking polls (``settle_timeout_s`` each) to
             catch frames that are still in flight from the most recent write burst.
        """
        if bus is None:
            return 0
        handler = getattr(bus, "channel_handler", None)
        if handler is None:
            return 0
        drained = 0

        # Phase 1: greedy non-blocking drain.
        for _ in range(max_frames):
            try:
                frame = handler.recv(timeout=0.0)
            except Exception:
                break
            if frame is None:
                break
            drained += 1

        # Phase 2: short blocking polls to catch in-flight frames from recent writes.
        for _ in range(max(0, settle_passes)):
            try:
                frame = handler.recv(timeout=settle_timeout_s)
            except Exception:
                break
            if frame is None:
                break
            drained += 1
            # Continue greedy after finding one — more may have queued up.
            for _ in range(max_frames):
                try:
                    f2 = handler.recv(timeout=0.0)
                except Exception:
                    break
                if f2 is None:
                    break
                drained += 1

        with self._read_stats_lock:
            self._rx_frames_drained += drained
        return drained

    def _read_bus_joints(
        self,
        bus,
        motors: list[tuple[str, int]],
        *,
        drain_rx: bool,
    ) -> np.ndarray:
        """Read one arm half (8 joints) from a single CAN bus."""
        out = np.zeros(len(motors), dtype=np.float64)
        if bus is None or not motors:
            return out
        if drain_rx:
            self._drain_bus_rx(bus)
        for i, (name, mid) in enumerate(motors):
            raw = self._read_one_with_retry(bus, name) * MOTOR_DIRECTION.get(mid, 1.0)
            out[i] = _normalize_near_zero(raw, MOTOR_SOFTWARE_ZERO.get(mid, 0.0))
        return out

    def _read_one_with_retry(self, bus, name: str, *, max_retries: int | None = None) -> float:
        """
        Read MECHANICAL_POSITION for a single motor with self-healing retry.

        Vendor ``bus.read()`` blocks reading the next frame and asserts it's a
        READ_PARAMETER reply.  If a stale OPERATION_STATUS arrives first, the assertion
        fails and the actual read reply gets stuck behind it.  We catch that, peel one
        stale frame off the RX queue (one retry per stale frame), and try again.
        """
        retries = self._read_max_retries if max_retries is None else max(0, int(max_retries))
        with self._read_stats_lock:
            self._read_attempts += 1
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return float(bus.read(name, self._ParameterType.MECHANICAL_POSITION))
            except Exception as exc:
                last_exc = exc
                # Peel one frame off — the next iteration's read will then have the
                # correct response (the one our previous transmit asked for) at the
                # queue head, OR another stale frame which we will peel again.
                handler = getattr(bus, "channel_handler", None)
                if handler is None:
                    break
                try:
                    handler.recv(timeout=0.002)
                except Exception:
                    pass
                with self._read_stats_lock:
                    self._rx_frames_drained += 1
        with self._read_stats_lock:
            self._read_failures += 1
        _ = last_exc  # kept for future debugging
        return 0.0

    def read_joints(self, *, drain_rx: bool = True) -> np.ndarray:
        """
        Read MECHANICAL_POSITION for all 16 joints.  Failed reads stay 0.0.

        Args:
            drain_rx: If True (default), flush stale frames from each bus's RX queue
                      before issuing register reads.  Strongly recommended when called
                      after ``command_joints`` on the same bus.
        """
        if not self._connected:
            raise RuntimeError("[move_actuators] Not connected.")
        with self._read_stats_lock:
            self._read_calls += 1
        out = np.zeros(NUM_JOINTS, dtype=np.float64)

        use_parallel = (
            self._parallel_bus_reads
            and self._left_bus is not None
            and self._right_bus is not None
            and self._left_motors
            and self._right_motors
        )
        if use_parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_left = pool.submit(
                    self._read_bus_joints,
                    self._left_bus,
                    self._left_motors,
                    drain_rx=drain_rx,
                )
                f_right = pool.submit(
                    self._read_bus_joints,
                    self._right_bus,
                    self._right_motors,
                    drain_rx=drain_rx,
                )
                left = f_left.result()
                right = f_right.result()
            out[:ARM_DOF] = left
            out[ARM_DOF:] = right
        else:
            if self._left_bus is not None:
                out[:ARM_DOF] = self._read_bus_joints(
                    self._left_bus, self._left_motors, drain_rx=drain_rx
                )
            if self._right_bus is not None:
                out[ARM_DOF:] = self._read_bus_joints(
                    self._right_bus, self._right_motors, drain_rx=drain_rx
                )

        return out

    # Backward-compatible alias.
    read_joints12 = read_joints

    def read_stats_line(self) -> str:
        """One-line summary of read reliability since connect()."""
        bad_pct = 100.0 * self._read_failures / max(self._read_attempts, 1)
        return (
            f"[move_actuators] read stats: calls={self._read_calls} "
            f"attempts={self._read_attempts} failures={self._read_failures} "
            f"({bad_pct:.1f}%) rx_frames_drained={self._rx_frames_drained}"
        )


# ---------------------------------------------------------------------------
# One-shot helper
# ---------------------------------------------------------------------------

def command_joints(
    angles12: np.ndarray | list | tuple,
    *,
    ramp: bool = True,
    ramp_max_speed_rad_s: float = RAMP_MAX_SPEED_RAD_S,
) -> np.ndarray:
    """Connect, send one command, disconnect.  Use ``ActuatorController`` for loops."""
    with ActuatorController(ramp=ramp, ramp_max_speed_rad_s=ramp_max_speed_rad_s) as arm:
        return arm.command_joints(angles12)


# Backward-compatible alias.
command_joints12 = command_joints


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_joints12(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != NUM_JOINTS:
        raise argparse.ArgumentTypeError(
            f"Expected {NUM_JOINTS} comma-separated floats, got {len(parts)}"
        )
    return np.array([float(x) for x in parts], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Send a single 16-DoF joint command to the RobStride arms.")
    p.add_argument(
        "--joints", type=_parse_joints12, required=True,
        metavar="q0,q1,...,q15",
        help="16 comma-separated joint angles in radians (L0..L6,Lg then R0..R6,Rg).",
    )
    p.add_argument("--no-ramp", action="store_true", help="Skip slew limiting (jump directly to targets).")
    p.add_argument("--hold-s", type=float, default=1.0, help="Hold position for this many seconds before exit.")
    args = p.parse_args()

    with ActuatorController(ramp=not args.no_ramp) as arm:
        sent = arm.command_joints(args.joints)
        labels = (
            "L0", "L1", "L2", "L3", "L4", "L5", "L6", "Lg",
            "R0", "R1", "R2", "R3", "R4", "R5", "R6", "Rg",
        )
        print("Commanded joints:")
        for j, (lab, v) in enumerate(zip(labels, sent)):
            print(f"  [{j:2d}] {lab}  {v:.4f} rad")
        if args.hold_s > 0:
            print(f"Holding for {args.hold_s}s ...")
            time.sleep(args.hold_s)


if __name__ == "__main__":
    main()