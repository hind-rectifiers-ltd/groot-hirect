"""
Independent module: drive 12 RobStride follower arm joints from a position vector.

This file has no dependency on any other file in this repo.  It can be imported
from any script.

Joint vector order (radians):
  [0..5]  left arm  — CAN ``can1``, motor IDs 1, 3, 5, 7, 9, 11
  [6..11] right arm — CAN ``can0``, motor IDs 2, 4, 6, 8, 10, 12

Typical use::

    from move_actuators import ActuatorController
    import numpy as np

    targets = np.zeros(12)
    targets[2] = 0.5   # left joint 2
    targets[9] = -0.3  # right joint 3

    with ActuatorController() as arm:
        # inside a control loop:
        sent = arm.command_joints12(targets)

One-shot helper (connects → sends → disconnects)::

    from move_actuators import command_joints12
    command_joints12([0.0] * 12)

CLI::

    uv run python move_actuators.py --joints 0,0,0,0,0,0,0,0,0,0,0,0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants (same values as direct_teleop.py)
# ---------------------------------------------------------------------------

LEFT_ROBSTRIDE_IDS: list[int] = [1, 3, 5, 7, 9, 11]
RIGHT_ROBSTRIDE_IDS: list[int] = [2, 4, 6, 8, 10, 12]
LEFT_CAN = "can1"
RIGHT_CAN = "can0"

MOTOR_MODEL_MAP: dict[int, str] = {
    1: "rs-03", 2: "rs-03", 3: "rs-03", 4: "rs-03",
    5: "rs-06", 6: "rs-06", 7: "rs-06", 8: "rs-06",
    9: "rs-02", 10: "rs-02", 11: "rs-02", 12: "rs-02",
}
MOTOR_KP: dict[int, float] = {
    1: 180.0, 2: 180.0, 3: 180.0, 4: 180.0, 5: 100.0, 6: 180.0,
    7: 180.0, 8: 180.0, 9: 30.0,  10: 30.0, 11: 30.0, 12: 30.0,
}
MOTOR_KD: dict[int, float] = {
    1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 18.0, 6: 18.0,
    7: 50.0, 8: 50.0, 9: 18.0, 10: 18.0, 11: 30.0, 12: 30.0,
}
MOTOR_TORQUE_LIMIT: dict[int, float] = {
    1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 12.0, 6: 12.0,
    7: 12.0, 8: 12.0, 9: 8.0,  10: 8.0,  11: 8.0, 12: 8.0,
}

RAMP_MAX_SPEED_RAD_S = 6.0   # rad/s slew limit (per joint, per second)
RAMP_DT_MAX_S = 0.1          # cap on dt used for ramp step calculation

NUM_JOINTS = 12

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


def _ramp_toward(current: float, desired: float, max_step: float) -> float:
    err = desired - current
    if err > max_step:
        return current + max_step
    if err < -max_step:
        return current - max_step
    return desired


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

    Open once, call ``command_joints12`` as fast as your loop runs,
    close when done.  Safe to use as a context manager::

        with ActuatorController() as arm:
            for targets in trajectory:
                arm.command_joints12(targets)
    """

    def __init__(
        self,
        *,
        ramp: bool = True,
        ramp_max_speed_rad_s: float = RAMP_MAX_SPEED_RAD_S,
        ramp_dt_max_s: float = RAMP_DT_MAX_S,
    ):
        """
        Args:
            ramp: Slew-limit each joint toward the target (recommended; prevents jerks).
            ramp_max_speed_rad_s: Maximum joint speed allowed by the ramp (rad/s).
            ramp_dt_max_s: dt is capped at this value when computing ramp step.
        """
        self._ramp = bool(ramp)
        self._ramp_max_speed = float(ramp_max_speed_rad_s)
        self._ramp_dt_max = float(ramp_dt_max_s)

        self._left_bus = None
        self._right_bus = None
        self._left_motors: list[tuple[str, int]] = []
        self._right_motors: list[tuple[str, int]] = []
        self._ramped: dict[str, float] = {}
        self._last_cmd_t: float | None = None
        self._connected = False

        # Read stats — useful when MIT writes and register reads compete for the bus.
        self._read_calls = 0       # number of times read_joints12() was invoked
        self._read_attempts = 0    # per-joint attempts (12 per call when both buses are live)
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
        self._connected = False

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

    def command_joints12(
        self,
        angles12: np.ndarray | list | tuple,
        *,
        ramp: bool | None = None,
    ) -> np.ndarray:
        """
        Send MIT position targets to all 12 joints.

        Args:
            angles12: Target joint angles in radians.  Length must be 12.
                      Order: [L0, L1, L2, L3, L4, Lg, R0, R1, R2, R3, R4, Rg].
            ramp: Override the instance-level ramp setting for this call only.

        Returns:
            The 12 targets actually written to the motors (after ramp limiting).

        Raises:
            RuntimeError: if not connected.
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

        for i, (name, mid) in enumerate(self._left_motors):
            desired = float(q[i])
            target = _ramp_toward(self._ramped.get(name, desired), desired, max_step)
            self._ramped[name] = target
            sent[i] = target
            if self._left_bus:
                try:
                    self._left_bus.write_operation_frame(
                        name, target, MOTOR_KP[mid], MOTOR_KD[mid], 0.0, 0.0
                    )
                except Exception:
                    pass

        for i, (name, mid) in enumerate(self._right_motors):
            desired = float(q[6 + i])
            target = _ramp_toward(self._ramped.get(name, desired), desired, max_step)
            self._ramped[name] = target
            sent[6 + i] = target
            if self._right_bus:
                try:
                    self._right_bus.write_operation_frame(
                        name, target, MOTOR_KP[mid], MOTOR_KD[mid], 0.0, 0.0
                    )
                except Exception:
                    pass

        return sent

    def seed_ramp_from_angles(self, angles12: np.ndarray | list | tuple) -> None:
        """
        Pre-load the internal ramp state with the given joint angles.

        After calling this, the next ``command_joints12`` will ramp *from* these
        angles toward the requested target instead of snapping straight to it.
        Typical use: read current encoder positions and call this before the
        first command of an inference / playback loop so the arm does not jerk
        from a stale (or zero) ramp seed.

        Args:
            angles12: 12-vector of seed angles, ordered like ``command_joints12``.
        """
        q = _pad12(angles12)
        for i, (name, _mid) in enumerate(self._left_motors):
            self._ramped[name] = float(q[i])
        for i, (name, _mid) in enumerate(self._right_motors):
            self._ramped[name] = float(q[6 + i])
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

        self._rx_frames_drained += drained
        return drained

    def _read_one_with_retry(self, bus, name: str, *, max_retries: int = 2) -> float:
        """
        Read MECHANICAL_POSITION for a single motor with self-healing retry.

        Vendor ``bus.read()`` blocks reading the next frame and asserts it's a
        READ_PARAMETER reply.  If a stale OPERATION_STATUS arrives first, the assertion
        fails and the actual read reply gets stuck behind it.  We catch that, peel one
        stale frame off the RX queue (one retry per stale frame), and try again.
        """
        self._read_attempts += 1
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
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
                self._rx_frames_drained += 1
        self._read_failures += 1
        _ = last_exc  # kept for future debugging
        return 0.0

    def read_joints12(self, *, drain_rx: bool = True) -> np.ndarray:
        """
        Read MECHANICAL_POSITION for all 12 joints.  Failed reads stay 0.0.

        Args:
            drain_rx: If True (default), flush stale frames from each bus's RX queue
                      before issuing register reads.  Strongly recommended when called
                      after ``command_joints12`` on the same bus.
        """
        if not self._connected:
            raise RuntimeError("[move_actuators] Not connected.")
        self._read_calls += 1
        out = np.zeros(NUM_JOINTS, dtype=np.float64)

        if drain_rx:
            self._drain_bus_rx(self._left_bus)
        if self._left_bus is not None:
            for i, (name, _) in enumerate(self._left_motors):
                out[i] = self._read_one_with_retry(self._left_bus, name)

        if drain_rx:
            self._drain_bus_rx(self._right_bus)
        if self._right_bus is not None:
            for i, (name, _) in enumerate(self._right_motors):
                out[6 + i] = self._read_one_with_retry(self._right_bus, name)

        return out

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

def command_joints12(
    angles12: np.ndarray | list | tuple,
    *,
    ramp: bool = True,
    ramp_max_speed_rad_s: float = RAMP_MAX_SPEED_RAD_S,
) -> np.ndarray:
    """Connect, send one command, disconnect.  Use ``ActuatorController`` for loops."""
    with ActuatorController(ramp=ramp, ramp_max_speed_rad_s=ramp_max_speed_rad_s) as arm:
        return arm.command_joints12(angles12)


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
    p = argparse.ArgumentParser(description="Send a single 12-DoF joint command to the RobStride arms.")
    p.add_argument(
        "--joints", type=_parse_joints12, required=True,
        metavar="q0,q1,...,q11",
        help="12 comma-separated joint angles in radians (L0..Lg then R0..Rg).",
    )
    p.add_argument("--no-ramp", action="store_true", help="Skip slew limiting (jump directly to targets).")
    p.add_argument("--hold-s", type=float, default=1.0, help="Hold position for this many seconds before exit.")
    args = p.parse_args()

    with ActuatorController(ramp=not args.no_ramp) as arm:
        sent = arm.command_joints12(args.joints)
        labels = ("L0", "L1", "L2", "L3", "L4", "Lg", "R0", "R1", "R2", "R3", "R4", "Rg")
        print("Commanded joints:")
        for j, (lab, v) in enumerate(zip(labels, sent)):
            print(f"  [{j:2d}] {lab}  {v:.4f} rad")
        if args.hold_s > 0:
            print(f"Holding for {args.hold_s}s ...")
            time.sleep(args.hold_s)


if __name__ == "__main__":
    main()
