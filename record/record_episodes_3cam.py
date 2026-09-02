#!/usr/bin/env python3
"""
Record per-episode HDF5 demos for a 3-camera humanoid setup.

Session mode (TTY): connect once, keep teleop running across episodes.
  r     start recording (even while a previous episode is saving)
  s/Esc stop current episode and save (background; waits only if a prior save is still running)
  d     discard current episode (no save; same episode index for next r)
  q     quit session (waits for any in-flight save, then disconnects)

Ctrl+C force-stops the current episode (saves if any frames), then exits the session.
If stdin is not a TTY, records a single episode until --max-steps or Ctrl+C.

Expected camera names:
  - cam_head
  - cam_left_wrist
  - cam_right_wrist

Expected state/action ordering (default 16D, matching direct_teleop):
  [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

Leader is still 5+1 Dynamixels per arm (12 total). Missing follower wrist_roll /
wrist_yaw joints are held at the teleop-zero follower refs (see
``direct_teleop.leader12_to_follower16``) — not absolute encoder 0.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class QposReadSanitizer:
    """
    Hold last-good qpos when a CAN read is clearly invalid.

    Catches:
      - failed reads that become 0.0 (move_actuators fallback)
      - single-frame garbage/jump values far from the previous good sample
    """

    def __init__(
        self,
        *,
        zero_epsilon: float = 0.05,
        last_good_min_rad: float = 0.12,
        max_step_rad: float = 0.45,
        max_hold_ticks: int = 12,
        state_dim: int = 16,
    ):
        self._zero_epsilon = float(zero_epsilon)
        self._last_good_min_rad = float(last_good_min_rad)
        self._max_step_rad = float(max(0.0, max_step_rad))
        self._max_hold_ticks = int(max(1, max_hold_ticks))
        self._state_dim = max(1, int(state_dim))
        self._last_good: np.ndarray | None = None
        self._hold_streak = np.zeros(self._state_dim, dtype=np.int32)
        self._hold_events = 0
        self._zero_holds = 0
        self._jump_holds = 0

    @staticmethod
    def max_step_for_rate(control_rate_hz: float, *, ramp_speed_rad_s: float = 6.0) -> float:
        """Per-tick jump limit from teleop rate and ramp speed (with margin)."""
        hz = max(control_rate_hz, 1e-3)
        return max(0.3, (float(ramp_speed_rad_s) / hz) * 1.25)

    def reset(self, qpos: np.ndarray) -> None:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size < self._state_dim:
            q = np.pad(q, (0, self._state_dim - q.size))
        self._last_good = q[: self._state_dim].copy()
        self._hold_streak[:] = 0

    @property
    def hold_events(self) -> int:
        return int(self._hold_events)

    @property
    def zero_holds(self) -> int:
        return int(self._zero_holds)

    @property
    def jump_holds(self) -> int:
        return int(self._jump_holds)

    def apply(self, qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size < self._state_dim:
            q = np.pad(q, (0, self._state_dim - q.size))
        q = q[: self._state_dim].copy()
        if self._last_good is None:
            self.reset(q)
            return q

        out = q.copy()
        for i in range(self._state_dim):
            v = float(q[i])
            prev = float(self._last_good[i])
            bad_zero, bad_jump = self._bad_sample_reasons(v, prev)
            if bad_zero or bad_jump:
                if self._hold_streak[i] < self._max_hold_ticks:
                    out[i] = prev
                    self._hold_streak[i] += 1
                    self._hold_events += 1
                    if bad_zero:
                        self._zero_holds += 1
                    if bad_jump:
                        self._jump_holds += 1
                    continue
            self._hold_streak[i] = 0
            self._last_good[i] = v
        return out

    @staticmethod
    def _shortest_delta_rad(value: float, last_good: float) -> float:
        d = float(value) - float(last_good)
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    def _bad_sample_reasons(self, value: float, last_good: float) -> tuple[bool, bool]:
        bad_zero = abs(value) <= self._zero_epsilon and abs(last_good) > self._last_good_min_rad
        bad_jump = False
        if self._max_step_rad > 0.0:
            bad_jump = abs(self._shortest_delta_rad(value, last_good)) > self._max_step_rad
        return bad_zero, bad_jump


def test_cameras(max_index: int, image_shape: tuple[int, int], warmup: int = 12, black_threshold: float = 5.0) -> None:
    """Probe video indices and report which return non-black frames (legacy numeric scan)."""
    from usb_cameras import open_capture, print_working_cameras

    if max_index <= 0:
        print_working_cameras()
        return

    h, w = image_shape
    working = []
    for i in range(max_index):
        path = f"/dev/video{i}"
        cap = open_capture(path)
        if not cap.isOpened():
            print(f"{path}: not opened")
            continue
        import cv2

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        mean_val = 0.0
        for _ in range(warmup):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                mean_val = float(np.mean(frame))
                if mean_val > black_threshold:
                    break
        cap.release()
        if mean_val > black_threshold:
            working.append(i)
            print(f"{path}: OK (mean pixel {mean_val:.1f})")
        else:
            print(f"{path}: black/no frames (mean {mean_val:.1f})")
    print(f"\nWorking indices: {working}")
    print("Tip: use --list-cameras / --list-cameras-working for stable USB port IDs.")


class RobotInterface:
    """Implement this interface for your real robot backend."""

    state_dim: int
    action_dim: int
    image_shape: tuple[int, int]  # (H, W)

    def get_observation(self) -> dict[str, Any]:
        """
        Must return:
          {
            "images": {
              "cam_head": uint8(H,W,3) RGB,
              "cam_left_wrist": uint8(H,W,3) RGB,
              "cam_right_wrist": uint8(H,W,3) RGB,
            },
            "qpos": float(state_dim),
            "qvel": float(state_dim)   # optional
            "effort": float(state_dim) # optional
          }
        """
        raise NotImplementedError

    def get_action(self) -> np.ndarray:
        """Return action float(action_dim)."""
        raise NotImplementedError

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass


class DemoRobotInterface(RobotInterface):
    def __init__(self, state_dim: int = 16, action_dim: int = 16, image_shape: tuple[int, int] = (240, 424)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_shape = image_shape

    def get_observation(self) -> dict[str, Any]:
        h, w = self.image_shape
        return {
            "images": {
                cam: np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for cam in CAMERA_NAMES
            },
            "qpos": (0.1 * np.random.randn(self.state_dim)).astype(np.float32),
            "qvel": np.zeros(self.state_dim, dtype=np.float32),
            "effort": np.zeros(self.state_dim, dtype=np.float32),
        }

    def get_action(self) -> np.ndarray:
        return (0.01 * np.random.randn(self.action_dim)).astype(np.float32)


class USBVideoRobotInterface(RobotInterface):
    """Camera-only backend. State/action are zeros unless updated externally."""

    def __init__(
        self,
        camera_devices: dict[str, int | str],
        state_dim: int = 16,
        action_dim: int = 16,
        image_shape: tuple[int, int] = (240, 424),
        camera_fps: float = 30.0,
    ):
        from usb_cameras import USBCameraRig

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_shape = image_shape
        self._qpos = np.zeros(self.state_dim, dtype=np.float32)
        self._action = np.zeros(self.action_dim, dtype=np.float32)
        self._camera_rig = USBCameraRig(camera_devices, image_shape, fps=camera_fps)

    def set_state_action(self, qpos: np.ndarray | None = None, action: np.ndarray | None = None) -> None:
        if qpos is not None:
            q = np.asarray(qpos, dtype=np.float32).reshape(-1)
            self._qpos[:] = q[: self.state_dim]
        if action is not None:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            self._action[:] = a[: self.action_dim]

    def get_observation(self) -> dict[str, Any]:
        images = self._camera_rig.read_all_rgb(CAMERA_NAMES)
        return {
            "images": images,
            "qpos": self._qpos.copy(),
            "qvel": np.zeros(self.state_dim, dtype=np.float32),
            "effort": np.zeros(self.state_dim, dtype=np.float32),
        }

    def get_action(self) -> np.ndarray:
        return self._action.copy()

    def close(self) -> None:
        self._camera_rig.close()


class DirectTeleopRobotInterface(USBVideoRobotInterface):
    """
    3-camera backend + direct local teleop loop (leader Dynamixel -> follower RobStride).

    Layout matches ``direct_teleop.py``:
      - Leader: 5 arm + 1 gripper per side (12 Dynamixels)
      - Follower: 7 arm + 1 gripper per side (16 RobStride); wrist_roll/yaw held at teleop-zero refs

    Motor I/O is delegated to ``move_actuators.ActuatorController`` (ramp + safety clamp).

    Records:
      - qpos: follower mechanical positions (16D)
      - action: ramp/clamp-limited commanded follower targets (16D)
    """

    def __init__(
        self,
        camera_devices: dict[str, int | str],
        leader_port: str = "/dev/ttyACM0",
        leader_baud: int = 57600,
        control_rate: float = 10.0,
        qpos_median_samples: int | None = None,
        qpos_median_gap_s: float = 0.003,
        dry_run: bool = False,
        safety_clamp: bool = True,
        image_shape: tuple[int, int] = (240, 424),
        camera_fps: float = 30.0,
    ):
        super().__init__(
            camera_devices=camera_devices,
            state_dim=16,
            action_dim=16,
            image_shape=image_shape,
            camera_fps=camera_fps,
        )
        self._repo_root = Path(__file__).resolve().parent.parent
        if str(self._repo_root) not in sys.path:
            sys.path.insert(0, str(self._repo_root))
        import direct_teleop as dt
        from move_actuators import ActuatorController, NUM_JOINTS, SafetyLimitBreachError

        self._dt = dt
        self._SafetyLimitBreachError = SafetyLimitBreachError
        self._num_joints = int(NUM_JOINTS)
        self._leader_num = int(dt.LEADER_NUM_JOINTS)
        dt.ensure_import_paths(self._repo_root)

        from dynamixel_easy_sdk import Connector

        self._connector = Connector(leader_port, leader_baud)
        self._leader_motors = self._connector.createAllMotors()
        if not self._leader_motors:
            raise RuntimeError("No leader Dynamixel motors found")
        for m in self._leader_motors:
            try:
                m.disableTorque()
                time.sleep(0.01)
            except Exception:
                pass

        self._dry_run = bool(dry_run)
        self._safety_clamp = bool(safety_clamp)
        self._left_motors_meta = [(f"motor_{mid}", mid) for mid in dt.LEFT_ROBSTRIDE_IDS]
        self._right_motors_meta = [(f"motor_{mid}", mid) for mid in dt.RIGHT_ROBSTRIDE_IDS]
        self._motor_ids: list[int] = list(dt.LEFT_ROBSTRIDE_IDS) + list(dt.RIGHT_ROBSTRIDE_IDS)
        self._control_rate_hz = float(control_rate)
        if qpos_median_samples is None:
            # 2 back-to-back reads reject single-frame CAN spikes (~22 ms/tick at 30 Hz).
            if self._control_rate_hz >= 20.0:
                qpos_median_samples = 2
            else:
                qpos_median_samples = 3
        self._qpos_median_samples = max(1, int(qpos_median_samples))
        # No sleep between median samples at high rate — gap only costs latency.
        if qpos_median_gap_s == 0.003 and self._control_rate_hz >= 20.0:
            qpos_median_gap_s = 0.0
        self._qpos_median_gap_s = max(0.0, float(qpos_median_gap_s))
        # At 30 Hz the per-tick jump filter false-triggers on real teleop motion; keep zero-dropout only.
        jump_limit = (
            0.0
            if self._control_rate_hz >= 20.0
            else QposReadSanitizer.max_step_for_rate(
                self._control_rate_hz,
                ramp_speed_rad_s=dt.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S,
            )
        )
        self._qpos_sanitizer = QposReadSanitizer(
            max_step_rad=jump_limit,
            state_dim=self._num_joints,
        )

        self._arm: ActuatorController | None = None
        if not self._dry_run:
            self._arm = ActuatorController(
                ramp=True,
                ramp_max_speed_rad_s=dt.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S,
                ramp_dt_max_s=dt.ROBSTRIDE_RAMP_DT_MAX_S,
                safety_clamp=self._safety_clamp,
                safety_abort_on_breach=not self._safety_clamp,
                read_max_retries=4,
                parallel_bus_reads=True,
            )
            self._arm.connect()
            # connect() already captures start-pose offsets (current pose = logical 0).
            encoders = self._arm.read_joints(samples=self._qpos_median_samples)
            self._arm.seed_ramp_from_angles(encoders)
            # Pre-flight: refuse to record unless the follower is already at the
            # start pose (logical ~0 after start-pose offset). Ensures the recorded
            # trajectory starts from a known zero without snap-from-stale-ramp noise.
            try:
                self._verify_zero_pose()
            except Exception:
                # Tear down everything opened so far so the process exits clean.
                self._cleanup_partial_init()
                raise

        self._qpos16 = np.zeros(self._num_joints, dtype=np.float32)
        self._action16 = np.zeros(self._num_joints, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._teleop_loop,
            kwargs={"control_rate": control_rate},
            daemon=True,
        )
        self._thread.start()
        print(
            f"[record] Teleop layout: leader {self._leader_num}D (5+1/arm) -> "
            f"follower {self._num_joints}D (7+1/arm); safety_clamp={self._safety_clamp}",
            flush=True,
        )
        if self._control_rate_hz >= 20.0:
            print(
                f"[record] High-rate qpos: median_samples={self._qpos_median_samples}, "
                f"gap_s={self._qpos_median_gap_s}, teleop_rate={control_rate} Hz "
                "(2-sample median + zero-dropout filter; jump filter off)",
                flush=True,
            )

    def _read_qpos16_raw(self, *, samples: int | None = None, sample_gap_s: float | None = None) -> np.ndarray:
        """
        Read follower pose via ``ActuatorController.read_joints`` (logical frame).

        Median / one-turn home unwrap live inside move_actuators — same path as safety.
        ``sample_gap_s`` is kept for API compatibility; high-rate record uses gap 0.
        """
        assert self._arm is not None
        n = max(1, int(samples if samples is not None else self._qpos_median_samples))
        _ = sample_gap_s  # median is internal to read_joints(samples=n)
        return self._arm.read_joints(samples=n)

    def _read_qpos16_for_command(self) -> tuple[np.ndarray, np.ndarray]:
        """
        One logical-frame read for logging and for ``command_joints(feedback12=...)``.

        Same vector for both — avoids record vs safety frame mismatch.
        """
        assert self._arm is not None
        q = self._arm.read_joints(samples=self._qpos_median_samples)
        qpos = self._qpos_sanitizer.apply(q)
        # Feedback uses the same logical read as qpos before hold-filter (safety frame).
        return qpos, q.copy()

    def _read_qpos16(self) -> np.ndarray:
        """Read qpos with median filtering and last-good hold for bad samples."""
        return self._qpos_sanitizer.apply(self._read_qpos16_raw())

    # Back-compat aliases used by older call sites / tools.
    _read_qpos12_raw = _read_qpos16_raw
    _read_qpos12_for_command = _read_qpos16_for_command
    _read_qpos12 = _read_qpos16

    def _verify_zero_pose(
        self,
        *,
        low: float = -0.2,
        high: float = 0.2,
        settle_reads: int = 3,
    ) -> None:
        """
        Pre-flight check: every follower motor must be at home (qpos in [low, high]).

        The first ``read_joints`` after ``connect`` may surface stale frames; we
        therefore read ``settle_reads`` times (drain + per-motor retry are already
        enabled inside ``ActuatorController``) and judge the last read.

        Raises RuntimeError naming each motor that is not at zero so the user knows
        exactly which arm joint to home before re-running.
        """
        assert self._arm is not None, "ActuatorController must be connected"
        qpos = np.zeros(self._num_joints, dtype=np.float64)
        for _ in range(max(1, settle_reads)):
            qpos = self._read_qpos16_raw()

        out_of_range: list[tuple[int, float]] = []
        for i, mid in enumerate(self._motor_ids):
            v = float(qpos[i])
            if not (low <= v <= high):
                out_of_range.append((mid, v))

        if out_of_range:
            details = "\n".join(
                f"  motor id {mid} is not zero (qpos={v:+.4f} rad, allowed range [{low}, {high}])"
                for mid, v in out_of_range
            )
            raise RuntimeError(
                "Follower pre-teleop zero-pose check failed:\n"
                + details
                + "\nMove the follower arm(s) back to home position (~0 rad) and re-run."
            )

        self._qpos_sanitizer.reset(qpos)
        qpos_str = ", ".join(f"{float(v):+.4f}" for v in qpos)
        print(f"[record] Follower zero-pose check OK: qpos=[{qpos_str}]", flush=True)

    def _cleanup_partial_init(self) -> None:
        """Release any resources opened in __init__, used when a constructor check fails."""
        if self._arm is not None:
            try:
                self._arm.disconnect(send_zero=False)
            except Exception:
                pass
            self._arm = None
        try:
            self._connector.closePort()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass

    def _teleop_loop(self, control_rate: float) -> None:
        dt = self._dt
        period = 1.0 / max(control_rate, 1e-3)
        teleop_initialized = False
        prev_servo = [0.0] * self._leader_num
        accum = [0.0] * self._leader_num
        robstride_ref = np.zeros(self._num_joints, dtype=np.float64)

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                angles = dt.get_joint_angles_from_motors(self._leader_motors)
                a12 = dt.pad_leader12(angles)
                if len(angles) < self._leader_num:
                    time.sleep(period)
                    continue

                if not teleop_initialized:
                    if self._arm is not None:
                        # Logical frame from ActuatorController (same as qpos / safety).
                        robstride_ref = self._arm.read_joints(
                            samples=self._qpos_median_samples
                        ).astype(np.float64)
                        self._arm.seed_ramp_from_angles(robstride_ref)
                    prev_servo = list(a12)
                    teleop_initialized = True
                    # Seed published state with the initial follower pose.
                    with self._lock:
                        q0 = self._read_qpos16() if self._arm is not None else np.zeros(self._num_joints)
                        self._qpos16[:] = q0.astype(np.float32)
                        self._action16[:] = q0.astype(np.float32)
                    continue

                # Accumulate leader-side servo deltas (in raw servo units).
                for i in range(self._leader_num):
                    accum[i] += dt.shortest_delta_units(prev_servo[i], a12[i])
                    prev_servo[i] = a12[i]

                # Map leader 5+1/arm → follower 7+1/arm (wrists held at teleop-zero refs).
                targets = dt.leader12_to_follower16(
                    accum,
                    robstride_ref,
                    self._left_motors_meta,
                    self._right_motors_meta,
                )

                # Read qpos *before* commanding so MIT status frames do not corrupt the read.
                if self._arm is not None:
                    qpos, feedback = self._read_qpos16_for_command()
                    sent = self._arm.command_joints(targets, ramp=True, feedback12=feedback)
                    action = np.asarray(sent, dtype=np.float64)
                else:
                    sent = targets
                    qpos = np.zeros(self._num_joints, dtype=np.float64)
                    action = qpos

                with self._lock:
                    self._qpos16[:] = qpos.astype(np.float32)
                    self._action16[:] = action.astype(np.float32)

            except self._SafetyLimitBreachError as exc:
                print(f"[record] SAFETY ABORT: {exc}", flush=True)
                self._stop_event.set()
                break
            except Exception as exc:
                print(f"[record] teleop loop error: {exc}", flush=True)

            elapsed = time.monotonic() - t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        with self._lock:
            obs["qpos"] = self._qpos16.copy()
        obs["qvel"] = np.zeros(self._num_joints, dtype=np.float32)
        obs["effort"] = np.zeros(self._num_joints, dtype=np.float32)
        return obs

    def get_action(self) -> np.ndarray:
        with self._lock:
            return self._action16.copy()

    def close(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._arm is not None:
            try:
                print(self._arm.read_stats_line(), flush=True)
            except Exception:
                pass
            holds = self._qpos_sanitizer.hold_events
            if holds > 0:
                print(
                    f"[record] qpos sanitizer holds: total={holds} "
                    f"(zero_dropouts={self._qpos_sanitizer.zero_holds}, "
                    f"jump_rejects={self._qpos_sanitizer.jump_holds})",
                    flush=True,
                )
                print(
                    "[record] WARNING: held qpos pollutes training data — "
                    "re-record this episode if zero_dropouts > 0.",
                    flush=True,
                )
            try:
                self._arm.disconnect(send_zero=True)
            except Exception:
                pass
            self._arm = None
        try:
            self._connector.closePort()
        except Exception:
            pass
        super().close()


@dataclass
class SessionKeyEvents:
    """Keyboard signals for the multi-episode recording session."""

    start: threading.Event = field(default_factory=threading.Event)  # r
    stop_save: threading.Event = field(default_factory=threading.Event)  # s or Esc
    discard: threading.Event = field(default_factory=threading.Event)  # d
    quit: threading.Event = field(default_factory=threading.Event)  # q
    shutdown: threading.Event = field(default_factory=threading.Event)
    available: bool = False  # False when stdin is not a TTY / listener could not start

    def clear_transient(self) -> None:
        self.start.clear()
        self.stop_save.clear()
        self.discard.clear()


def _start_session_key_listener() -> tuple[SessionKeyEvents, Callable[[], None]]:
    """
    Background keyboard listener for the whole recording session.

    Keys (TTY):
      r / R  → start recording
      s / S  → stop + save
      Esc    → stop + save (same as s)
      d / D  → discard (no save)
      q / Q  → quit session

    Returns (events, join_fn). Caller must set events.shutdown and call join_fn on exit.
    """
    events = SessionKeyEvents()

    def join_timeout() -> None:
        events.shutdown.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    thread: threading.Thread | None = None

    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return events, join_timeout

        def _win_loop() -> None:
            events.available = True
            while not events.shutdown.is_set():
                if msvcrt.kbhit():
                    c = msvcrt.getch()
                    if c in (b"r", b"R"):
                        events.start.set()
                    elif c in (b"s", b"S") or c == b"\x1b":
                        events.stop_save.set()
                    elif c in (b"d", b"D"):
                        events.discard.set()
                    elif c in (b"q", b"Q"):
                        events.quit.set()
                time.sleep(0.02)

        thread = threading.Thread(target=_win_loop, daemon=True)
        thread.start()
        return events, join_timeout

    import select
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
    except (OSError, ValueError):
        return events, join_timeout
    if not os.isatty(fd):
        return events, join_timeout

    def _posix_loop() -> None:
        old: list[Any] | None = None
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            events.available = True
            while not events.shutdown.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("r", "R"):
                    events.start.set()
                    continue
                if ch in ("s", "S"):
                    events.stop_save.set()
                    continue
                if ch in ("d", "D"):
                    events.discard.set()
                    continue
                if ch in ("q", "Q"):
                    events.quit.set()
                    continue
                if ch != "\x1b":
                    continue
                # Lone Esc vs CSI arrow sequences.
                r2, _, _ = select.select([sys.stdin], [], [], 0.04)
                if r2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        r3, _, _ = select.select([sys.stdin], [], [], 0.04)
                        if r3:
                            sys.stdin.read(1)
                    continue
                events.stop_save.set()
        except (OSError, termios.error, ValueError):
            pass
        finally:
            if old is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except (OSError, termios.error):
                    pass

    thread = threading.Thread(target=_posix_loop, daemon=True)
    thread.start()
    # Tiny settle so available flag is set before caller prints help.
    time.sleep(0.05)
    return events, join_timeout


def get_next_episode_index(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in output_dir.glob("episode_*.hdf5"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    return 0 if not ids else max(ids) + 1


@dataclass
class EpisodePackage:
    """Finished episode buffers handed off to the save worker (ownership transfers)."""

    output_path: Path
    task: str
    dt: float
    timestamps: list[float]
    images: dict[str, list[np.ndarray]]
    qpos: list[np.ndarray]
    actions: list[np.ndarray]
    qvel: list[np.ndarray]
    effort: list[np.ndarray]
    include_qvel: bool
    include_effort: bool


def write_episode_hdf5(pkg: EpisodePackage) -> None:
    """Write one episode to HDF5 (gzip images). Uses a temp file then rename."""
    output_path = pkg.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    n = len(pkg.timestamps)
    print(f"[save] Writing {output_path.name} ({n} steps)...", flush=True)
    t0 = time.monotonic()
    try:
        with h5py.File(tmp_path, "w") as root:
            root.attrs["task"] = pkg.task
            root.attrs["fps"] = 1.0 / pkg.dt if pkg.dt > 0 else 0.0
            root.create_dataset("/timestamp", data=np.asarray(pkg.timestamps, dtype=np.float64))
            obs_grp = root.create_group("observations")
            img_grp = obs_grp.create_group("images")
            for cam in CAMERA_NAMES:
                img_grp.create_dataset(
                    cam,
                    data=np.asarray(pkg.images[cam], dtype=np.uint8),
                    compression="gzip",
                )
            obs_grp.create_dataset("qpos", data=np.asarray(pkg.qpos, dtype=np.float32))
            root.create_dataset("action", data=np.asarray(pkg.actions, dtype=np.float32))
            if pkg.include_qvel and pkg.qvel:
                obs_grp.create_dataset("qvel", data=np.asarray(pkg.qvel, dtype=np.float32))
            if pkg.include_effort and pkg.effort:
                obs_grp.create_dataset("effort", data=np.asarray(pkg.effort, dtype=np.float32))
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    elapsed = time.monotonic() - t0
    print(f"[save] Saved {output_path.name} in {elapsed:.1f}s.", flush=True)


class EpisodeSaveWorker:
    """
    At most one in-flight HDF5 save.

    ``submit`` blocks only when a save is already running (max 1). Recording the
    next episode does **not** wait — only finishing an episode and enqueueing its
    save may wait for the prior save to complete.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def busy(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def wait_until_idle(self, *, reason: str) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            print(f"[save] Waiting for previous episode save to finish ({reason})...", flush=True)
            t.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            err = self._error
            self._error = None
            raise RuntimeError(f"Background episode save failed: {err}") from err

    def submit(self, pkg: EpisodePackage) -> None:
        self.wait_until_idle(reason="before starting next save")
        self._error = None

        def _run() -> None:
            try:
                write_episode_hdf5(pkg)
            except BaseException as exc:  # noqa: BLE001 — surface to main via wait
                self._error = exc
                print(f"[save] ERROR writing {pkg.output_path.name}: {exc}", flush=True)

        with self._lock:
            self._thread = threading.Thread(
                target=_run,
                name=f"save-{pkg.output_path.name}",
                daemon=False,
            )
            self._thread.start()


def _capture_one_step(
    robot: RobotInterface,
    *,
    imgs: dict[str, list[np.ndarray]],
    obs_qpos: list[np.ndarray],
    act: list[np.ndarray],
    obs_qvel: list[np.ndarray],
    obs_effort: list[np.ndarray],
    ts: list[float],
    t0: float,
    include_qvel: bool,
    include_effort: bool,
) -> None:
    step_start = time.time()
    o = robot.get_observation()
    a = robot.get_action()
    for cam in CAMERA_NAMES:
        imgs[cam].append(np.asarray(o["images"][cam], dtype=np.uint8))
    obs_qpos.append(np.asarray(o["qpos"], dtype=np.float32))
    act.append(np.asarray(a, dtype=np.float32))
    if include_qvel and "qvel" in o:
        obs_qvel.append(np.asarray(o["qvel"], dtype=np.float32))
    if include_effort and "effort" in o:
        obs_effort.append(np.asarray(o["effort"], dtype=np.float32))
    ts.append(float(step_start - t0))


def _package_buffers(
    *,
    output_path: Path,
    task: str,
    dt: float,
    imgs: dict[str, list[np.ndarray]],
    obs_qpos: list[np.ndarray],
    act: list[np.ndarray],
    obs_qvel: list[np.ndarray],
    obs_effort: list[np.ndarray],
    ts: list[float],
    include_qvel: bool,
    include_effort: bool,
) -> EpisodePackage:
    return EpisodePackage(
        output_path=output_path,
        task=task,
        dt=dt,
        timestamps=ts,
        images=imgs,
        qpos=obs_qpos,
        actions=act,
        qvel=obs_qvel,
        effort=obs_effort,
        include_qvel=include_qvel,
        include_effort=include_effort,
    )


def run_recording_session(
    robot: RobotInterface,
    output_dir: Path,
    *,
    task: str,
    max_steps: int,
    dt: float,
    include_qvel: bool = True,
    include_effort: bool = True,
    start_episode_idx: int | None = None,
) -> None:
    """
    Multi-episode session: teleop stays connected; r/s/q (Esc=stop+save).

    Non-TTY: records a single episode until max_steps / Ctrl+C, then exits.
    """
    keys, keys_join = _start_session_key_listener()
    saver = EpisodeSaveWorker()
    ep_idx = start_episode_idx if start_episode_idx is not None else get_next_episode_index(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_shot = not keys.available
    if single_shot:
        print(
            "[session] stdin is not a TTY — recording one episode until --max-steps or Ctrl+C.",
            flush=True,
        )
    else:
        print(
            "[session] Teleop stays live across episodes.\n"
            "  r     start recording (can start while previous episode saves in background)\n"
            "  s/Esc stop + save (waits only if a prior save is still running)\n"
            "  d     discard current episode (no save; retry same episode index)\n"
            "  q     quit (waits for save, then disconnects)\n"
            "  Ctrl+C  stop current episode (save if any frames) and quit",
            flush=True,
        )

    quit_session = False
    try:
        while not quit_session:
            if single_shot:
                # Auto-start the only episode.
                keys.start.set()
            else:
                save_hint = " (background save in progress)" if saver.busy() else ""
                print(
                    f"\n[idle] Teleop live{save_hint}. "
                    f"Press r to record episode_{ep_idx:06d}.hdf5, q to quit.",
                    flush=True,
                )
                keys.clear_transient()
                while not keys.quit.is_set():
                    if keys.start.is_set():
                        keys.start.clear()
                        break
                    # Esc/s/d in idle are ignored (nothing to stop/discard).
                    if keys.stop_save.is_set():
                        keys.stop_save.clear()
                    if keys.discard.is_set():
                        keys.discard.clear()
                    time.sleep(0.05)
                if keys.quit.is_set():
                    break

            out = output_dir / f"episode_{ep_idx:06d}.hdf5"
            if saver.busy():
                print(
                    f"\n[record] Starting {out.name} while previous episode still saving...",
                    flush=True,
                )
            print(f"\n[record] Recording -> {out.name}  (s/Esc=save, d=discard, q=save+quit)", flush=True)
            robot.on_episode_start()
            obs_qpos: list[np.ndarray] = []
            act: list[np.ndarray] = []
            obs_qvel: list[np.ndarray] = []
            obs_effort: list[np.ndarray] = []
            imgs: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_NAMES}
            ts: list[float] = []
            t0 = time.time()
            n = 0
            stop_reason = "max_steps"
            keys.clear_transient()

            try:
                for _ in range(max_steps):
                    if keys.quit.is_set():
                        stop_reason = "quit"
                        break
                    if keys.discard.is_set():
                        keys.discard.clear()
                        stop_reason = "discard"
                        break
                    if keys.stop_save.is_set():
                        keys.stop_save.clear()
                        stop_reason = "stop"
                        break
                    step_start = time.time()
                    _capture_one_step(
                        robot,
                        imgs=imgs,
                        obs_qpos=obs_qpos,
                        act=act,
                        obs_qvel=obs_qvel,
                        obs_effort=obs_effort,
                        ts=ts,
                        t0=t0,
                        include_qvel=include_qvel,
                        include_effort=include_effort,
                    )
                    n += 1
                    if n == 1 or n % 30 == 0:
                        print(f"\r[record] steps={n}", end="", flush=True)
                    time.sleep(max(0.0, dt - (time.time() - step_start)))
            except KeyboardInterrupt:
                stop_reason = "interrupt"
                print("\n[record] Ctrl+C — stopping episode.", flush=True)
            finally:
                robot.on_episode_end()
                if n > 0:
                    print(flush=True)

            if n == 0:
                print("[record] No frames captured; not saving.", flush=True)
            elif stop_reason == "discard":
                print(f"[record] Discarded {out.name} ({n} steps, not saved).", flush=True)
                del imgs, obs_qpos, act, obs_qvel, obs_effort, ts
            else:
                reason_msg = {
                    "stop": "s/Esc",
                    "quit": "q",
                    "interrupt": "Ctrl+C",
                    "max_steps": "max-steps",
                }.get(stop_reason, stop_reason)
                print(f"[record] End ({reason_msg}), {n} steps → queue save {out.name}", flush=True)
                pkg = _package_buffers(
                    output_path=out,
                    task=task,
                    dt=dt,
                    imgs=imgs,
                    obs_qpos=obs_qpos,
                    act=act,
                    obs_qvel=obs_qvel,
                    obs_effort=obs_effort,
                    ts=ts,
                    include_qvel=include_qvel,
                    include_effort=include_effort,
                )
                # Drop local refs; save thread owns the arrays.
                del imgs, obs_qpos, act, obs_qvel, obs_effort, ts
                saver.submit(pkg)
                ep_idx += 1

            if stop_reason in ("quit", "interrupt") or single_shot:
                quit_session = True

    finally:
        try:
            saver.wait_until_idle(reason="before exit")
        except RuntimeError as exc:
            print(f"[session] {exc}", flush=True)
        keys.shutdown.set()
        keys_join()
        print("[session] Exiting. Disconnecting robot...", flush=True)
        if hasattr(robot, "close"):
            robot.close()


def record_episode(
    robot: RobotInterface,
    output_path: Path,
    task: str,
    max_steps: int,
    dt: float,
    include_qvel: bool = True,
    include_effort: bool = True,
) -> tuple[bool, bool]:
    """
    Record a single episode (legacy helper). Prefer ``run_recording_session`` for demos.

    Returns (saved_ok, interrupted_by_ctrl_c).
    """
    # Thin wrapper: one-shot session into a fixed path via a temp session dir trick
    # is awkward; keep a direct capture+save path for callers/tests.
    keys, keys_join = _start_session_key_listener()
    robot.on_episode_start()
    obs_qpos: list[np.ndarray] = []
    act: list[np.ndarray] = []
    obs_qvel: list[np.ndarray] = []
    obs_effort: list[np.ndarray] = []
    imgs: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_NAMES}
    ts: list[float] = []
    t0 = time.time()
    n = 0
    interrupted = False
    try:
        for _ in range(max_steps):
            if keys.stop_save.is_set() or keys.quit.is_set():
                print("\nEnd of episode (Esc/s/q). Saving frames captured so far...", flush=True)
                break
            step_start = time.time()
            _capture_one_step(
                robot,
                imgs=imgs,
                obs_qpos=obs_qpos,
                act=act,
                obs_qvel=obs_qvel,
                obs_effort=obs_effort,
                ts=ts,
                t0=t0,
                include_qvel=include_qvel,
                include_effort=include_effort,
            )
            n += 1
            time.sleep(max(0.0, dt - (time.time() - step_start)))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped early (Ctrl+C). Saving frames captured so far...", flush=True)
    finally:
        keys.shutdown.set()
        keys_join()
        robot.on_episode_end()

    if n == 0:
        return False, interrupted

    write_episode_hdf5(
        _package_buffers(
            output_path=output_path,
            task=task,
            dt=dt,
            imgs=imgs,
            obs_qpos=obs_qpos,
            act=act,
            obs_qvel=obs_qvel,
            obs_effort=obs_effort,
            ts=ts,
            include_qvel=include_qvel,
            include_effort=include_effort,
        )
    )
    return True, interrupted


def main() -> None:
    p = argparse.ArgumentParser(description="Record 3-camera episodes for GR00T-compatible conversion")
    from usb_cameras import add_three_camera_cli_args

    add_three_camera_cli_args(p)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--task", type=str, default="demo task")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--dt", type=float, default=1.0 / 30.0)
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--action-dim", type=int, default=16)
    p.add_argument("--image-height", type=int, default=640)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--robot", choices=["demo", "usb_cam", "direct_teleop"], default="demo")
    p.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    p.add_argument("--leader-baud", type=int, default=57600)
    p.add_argument("--teleop-rate", type=float, default=10.0)
    p.add_argument(
        "--qpos-median-samples",
        type=int,
        default=None,
        help="Per-tick encoder reads for qpos median (default: 2 if teleop-rate >=20, else 3)",
    )
    p.add_argument(
        "--qpos-median-gap-ms",
        type=float,
        default=3.0,
        help="Gap between median samples in ms (auto 0 at teleop-rate >=20)",
    )
    p.add_argument("--dry-run-teleop", action="store_true")
    p.add_argument(
        "--safety-abort",
        action="store_true",
        help="Abort+disconnect on safety breach instead of clamping targets (default: clamp).",
    )
    p.add_argument("--episode-idx", type=int, default=None)
    p.add_argument("--no-qvel", action="store_true")
    p.add_argument("--no-effort", action="store_true")
    p.add_argument("--test-cameras", action="store_true")
    p.add_argument("--test-cameras-max", type=int, default=12)
    args = p.parse_args()

    from usb_cameras import camera_cli_from_args, handle_camera_list_flags

    if handle_camera_list_flags(args):
        return

    if args.output_dir is None:
        p.error("--output-dir is required (unless using --list-cameras / --preview-cameras)")

    if args.test_cameras:
        test_cameras(
            max_index=args.test_cameras_max,
            image_shape=(args.image_height, args.image_width),
        )
        return

    if args.robot == "demo":
        robot: RobotInterface = DemoRobotInterface(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            image_shape=(args.image_height, args.image_width),
        )
    elif args.robot == "usb_cam":
        camera_devices = camera_cli_from_args(args)
        robot = USBVideoRobotInterface(
            camera_devices=camera_devices,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            image_shape=(args.image_height, args.image_width),
            camera_fps=float(args.camera_fps),
        )
    else:
        if args.state_dim != 16 or args.action_dim != 16:
            print(
                "WARNING: direct_teleop records 16D state/action (7+1 per arm). "
                f"Overriding dims to 16 (got state={args.state_dim}, action={args.action_dim}).",
                flush=True,
            )
            args.state_dim = 16
            args.action_dim = 16
        try:
            camera_devices = camera_cli_from_args(args)
            robot = DirectTeleopRobotInterface(
                camera_devices=camera_devices,
                leader_port=args.leader_port,
                leader_baud=args.leader_baud,
                control_rate=args.teleop_rate,
                qpos_median_samples=args.qpos_median_samples,
                qpos_median_gap_s=max(0.0, args.qpos_median_gap_ms / 1000.0),
                dry_run=args.dry_run_teleop,
                safety_clamp=not args.safety_abort,
                image_shape=(args.image_height, args.image_width),
                camera_fps=float(args.camera_fps),
            )
        except RuntimeError as exc:
            # Pre-flight zero-pose check (or other init validation) failed: print and exit
            # cleanly without recording anything.  Resources are already released by
            # DirectTeleopRobotInterface._cleanup_partial_init.
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            sys.exit(1)

    ep_idx = args.episode_idx if args.episode_idx is not None else get_next_episode_index(args.output_dir)
    print(f"[session] Output dir: {args.output_dir}  (next episode index: {ep_idx})", flush=True)
    print("Starting in 2 seconds (teleop already live if direct_teleop)...", flush=True)
    time.sleep(2)

    try:
        run_recording_session(
            robot,
            args.output_dir,
            task=args.task,
            max_steps=args.max_steps,
            dt=args.dt,
            include_qvel=not args.no_qvel,
            include_effort=not args.no_effort,
            start_episode_idx=ep_idx,
        )
    except Exception:
        # Session normally closes the robot in its finally; if connect failed earlier
        # or session raised before that, still try to release.
        if hasattr(robot, "close"):
            try:
                robot.close()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
