#!/usr/bin/env python3
"""
Record per-episode HDF5 demos for a 3-camera humanoid setup.

Press Esc in the recording terminal to stop and save the current episode (Ctrl+C still
force-stops and saves). If stdin is not a TTY, use --max-steps or Ctrl+C.

Expected camera names:
  - cam_head
  - cam_left_wrist
  - cam_right_wrist

Expected state/action ordering (default 12D):
  [left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _normalize_qpos_to_pi(qpos: np.ndarray) -> np.ndarray:
    """
    Wrap joint angles to [-pi, pi) for dataset storage.

    This is a final safety-net for wrapped encoder values (~2*pi at physical zero).
    Commanding still uses the motor-native frame elsewhere.
    """
    q = np.asarray(qpos, dtype=np.float64)
    return ((q + np.pi) % (2.0 * np.pi)) - np.pi


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
    ):
        self._zero_epsilon = float(zero_epsilon)
        self._last_good_min_rad = float(last_good_min_rad)
        self._max_step_rad = float(max(0.0, max_step_rad))
        self._max_hold_ticks = int(max(1, max_hold_ticks))
        self._last_good: np.ndarray | None = None
        self._hold_streak = np.zeros(12, dtype=np.int32)
        self._hold_events = 0
        self._zero_holds = 0
        self._jump_holds = 0

    @staticmethod
    def max_step_for_rate(control_rate_hz: float, *, ramp_speed_rad_s: float = 6.0) -> float:
        """Per-tick jump limit from teleop rate and ramp speed (with margin)."""
        hz = max(control_rate_hz, 1e-3)
        return max(0.3, (float(ramp_speed_rad_s) / hz) * 1.25)

    def reset(self, qpos12: np.ndarray) -> None:
        q = np.asarray(qpos12, dtype=np.float64).reshape(-1)
        if q.size < 12:
            q = np.pad(q, (0, 12 - q.size))
        self._last_good = q[:12].copy()
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

    def apply(self, qpos12: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos12, dtype=np.float64).reshape(-1)
        if q.size < 12:
            q = np.pad(q, (0, 12 - q.size))
        q = q[:12].copy()
        if self._last_good is None:
            self.reset(q)
            return q

        out = q.copy()
        for i in range(12):
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
    """Probe video indices and report which return non-black frames."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for --test-cameras") from exc

    h, w = image_shape
    working = []
    for i in range(max_index):
        path = f"/dev/video{i}"
        cap = cv2.VideoCapture(path, getattr(cv2, "CAP_V4L2", 200))
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"{path}: not opened")
            continue
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
    def __init__(self, state_dim: int = 12, action_dim: int = 12, image_shape: tuple[int, int] = (240, 424)):
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
        cam_head_device: int | str,
        cam_left_wrist_device: int | str,
        cam_right_wrist_device: int | str,
        state_dim: int = 12,
        action_dim: int = 12,
        image_shape: tuple[int, int] = (240, 424),
    ):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for USB camera mode") from exc

        self._cv2 = cv2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_shape = image_shape
        self._qpos = np.zeros(self.state_dim, dtype=np.float32)
        self._action = np.zeros(self.action_dim, dtype=np.float32)

        device_map: dict[str, int | str] = {
            "cam_head": cam_head_device,
            "cam_left_wrist": cam_left_wrist_device,
            "cam_right_wrist": cam_right_wrist_device,
        }
        self._caps = {}
        for cam_name, dev in device_map.items():
            path = f"/dev/video{dev}" if isinstance(dev, int) else str(dev)
            cap = cv2.VideoCapture(path, getattr(cv2, "CAP_V4L2", 200))
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {cam_name} at {path}")
            h, w = image_shape
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._caps[cam_name] = cap
            time.sleep(0.2)

    def set_state_action(self, qpos: np.ndarray | None = None, action: np.ndarray | None = None) -> None:
        if qpos is not None:
            q = np.asarray(qpos, dtype=np.float32).reshape(-1)
            self._qpos[:] = q[: self.state_dim]
        if action is not None:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            self._action[:] = a[: self.action_dim]

    def get_observation(self) -> dict[str, Any]:
        cv2 = self._cv2
        h, w = self.image_shape
        images = {}
        for cam in CAMERA_NAMES:
            ret, frame = self._caps[cam].read()
            if not ret or frame is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images[cam] = img.astype(np.uint8)

        return {
            "images": images,
            "qpos": self._qpos.copy(),
            "qvel": np.zeros(self.state_dim, dtype=np.float32),
            "effort": np.zeros(self.state_dim, dtype=np.float32),
        }

    def get_action(self) -> np.ndarray:
        return self._action.copy()

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()


class DirectTeleopRobotInterface(USBVideoRobotInterface):
    """
    3-camera backend + direct local teleop loop (leader Dynamixel -> follower RobStride).

    Motor I/O is delegated to ``move_actuators.ActuatorController``, which handles the
    write-then-drain-then-read pipeline with per-motor retry that was validated in
    ``direct_teleop.py``.  This eliminates the qpos-drop-to-zero failure mode caused by
    stale OPERATION_STATUS frames sitting in the python-can RX queue.

    Records:
      - qpos: follower mechanical positions (12D), read via ActuatorController.read_joints12
      - action: ramp-limited commanded follower targets (12D), produced by command_joints12
    """

    LEFT_SERVO_INDICES = [0, 2, 4, 6, 8, 10]
    RIGHT_SERVO_INDICES = [1, 3, 5, 7, 9, 11]

    def __init__(
        self,
        cam_head_device: int | str,
        cam_left_wrist_device: int | str,
        cam_right_wrist_device: int | str,
        leader_port: str = "/dev/ttyACM0",
        leader_baud: int = 57600,
        control_rate: float = 10.0,
        qpos_median_samples: int | None = None,
        qpos_median_gap_s: float = 0.003,
        dry_run: bool = False,
        image_shape: tuple[int, int] = (240, 424),
    ):
        super().__init__(
            cam_head_device=cam_head_device,
            cam_left_wrist_device=cam_left_wrist_device,
            cam_right_wrist_device=cam_right_wrist_device,
            state_dim=12,
            action_dim=12,
            image_shape=image_shape,
        )
        self._repo_root = Path(__file__).resolve().parent.parent
        if str(self._repo_root) not in sys.path:
            sys.path.insert(0, str(self._repo_root))
        import direct_teleop as dt
        from move_actuators import ActuatorController

        self._dt = dt
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
        self._left_motors_meta = [(f"motor_{mid}", mid) for mid in dt.LEFT_ROBSTRIDE_IDS]
        self._right_motors_meta = [(f"motor_{mid}", mid) for mid in dt.RIGHT_ROBSTRIDE_IDS]
        self._motor_ids: list[int] = list(dt.LEFT_ROBSTRIDE_IDS) + list(dt.RIGHT_ROBSTRIDE_IDS)
        # One-turn unwrap offsets in encoder frame. Reported qpos is (raw - offsets).
        self._boot_offsets = np.zeros(12, dtype=np.float64)
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
        self._qpos_sanitizer = QposReadSanitizer(max_step_rad=jump_limit)

        self._arm: ActuatorController | None = None
        if not self._dry_run:
            self._arm = ActuatorController(
                ramp=True,
                ramp_max_speed_rad_s=dt.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S,
                ramp_dt_max_s=dt.ROBSTRIDE_RAMP_DT_MAX_S,
                read_max_retries=4,
                parallel_bus_reads=True,
            )
            self._arm.connect()
            self._capture_boot_offsets()
            # Pre-flight: refuse to record unless the follower is already at home.
            # Ensures the recorded trajectory starts from a known zero pose so we
            # do not save ramp-from-arbitrary-pose noise as the first frames.
            try:
                self._verify_zero_pose()
            except Exception:
                # Tear down everything opened so far so the process exits clean.
                self._cleanup_partial_init()
                raise

        self._qpos12 = np.zeros(12, dtype=np.float32)
        self._action12 = np.zeros(12, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._teleop_loop,
            kwargs={"control_rate": control_rate},
            daemon=True,
        )
        self._thread.start()
        if self._control_rate_hz >= 20.0:
            print(
                f"[record] High-rate qpos: median_samples={self._qpos_median_samples}, "
                f"gap_s={self._qpos_median_gap_s}, teleop_rate={control_rate} Hz "
                "(2-sample median + zero-dropout filter; jump filter off)",
                flush=True,
            )

    def _capture_boot_offsets(self, *, settle_reads: int = 3) -> None:
        """Apply one-turn software unwrap so home near 0 is represented near 0."""
        assert self._arm is not None, "ActuatorController must be connected"
        raw = np.zeros(12, dtype=np.float64)
        for _ in range(max(1, settle_reads)):
            raw = self._arm.read_joints12().astype(np.float64)
        adjusted: list[tuple[int, float, float]] = []
        for i, mid in enumerate(self._motor_ids):
            v = float(raw[i])
            if v > np.pi:
                self._boot_offsets[i] = 2.0 * np.pi
            elif v < -np.pi:
                self._boot_offsets[i] = -2.0 * np.pi
            else:
                self._boot_offsets[i] = 0.0
            if self._boot_offsets[i] != 0.0:
                adjusted.append((mid, v, v - self._boot_offsets[i]))
        if adjusted:
            print("[record] Software zero: one-turn unwrap applied to:", flush=True)
            for mid, before, after in adjusted:
                print(f"  motor id {mid}: encoder={before:+.4f} rad -> reported as {after:+.4f} rad", flush=True)
        else:
            print("[record] Software zero: all joints already inside (-pi, +pi).", flush=True)

    def _read_qpos12_raw(self, *, samples: int | None = None, sample_gap_s: float | None = None) -> np.ndarray:
        """
        Read follower pose in unwrapped software-zero frame (no dropout filtering).

        Takes the per-joint median of ``samples`` reads to reject single garbage frames.
        """
        assert self._arm is not None
        n = max(1, int(samples if samples is not None else self._qpos_median_samples))
        gap = self._qpos_median_gap_s if sample_gap_s is None else max(0.0, float(sample_gap_s))
        stack = []
        for s in range(n):
            raw = self._arm.read_joints12().astype(np.float64)
            stack.append(raw - self._boot_offsets)
            if s + 1 < n and gap > 0.0:
                time.sleep(gap)
        return np.median(np.stack(stack, axis=0), axis=0)

    def _read_qpos12_for_command(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Read qpos for logging and return raw encoder feedback for the same tick.

        Returns:
            qpos: normalized software-zero frame for dataset logging.
            raw_encoder: native encoder frame for ``command_joints12(feedback12=...)``.
        """
        assert self._arm is not None
        n = self._qpos_median_samples
        gap = self._qpos_median_gap_s
        stack_sw: list[np.ndarray] = []
        stack_raw: list[np.ndarray] = []
        for s in range(n):
            raw = self._arm.read_joints12().astype(np.float64)
            stack_raw.append(raw)
            stack_sw.append(raw - self._boot_offsets)
            if s + 1 < n and gap > 0.0:
                time.sleep(gap)
        if n == 1:
            raw_encoder = stack_raw[0]
            sw = stack_sw[0]
        else:
            raw_encoder = np.median(np.stack(stack_raw, axis=0), axis=0)
            sw = np.median(np.stack(stack_sw, axis=0), axis=0)
        qpos = _normalize_qpos_to_pi(self._qpos_sanitizer.apply(sw))
        return qpos, raw_encoder

    def _read_qpos12(self) -> np.ndarray:
        """Read qpos with median filtering and last-good hold for bad samples."""
        q = self._qpos_sanitizer.apply(self._read_qpos12_raw())
        return _normalize_qpos_to_pi(q)

    def _verify_zero_pose(
        self,
        *,
        low: float = -0.2,
        high: float = 0.2,
        settle_reads: int = 3,
    ) -> None:
        """
        Pre-flight check: every follower motor must be at home (qpos in [low, high]).

        The first ``read_joints12`` after ``connect`` may surface stale frames; we
        therefore read ``settle_reads`` times (drain + per-motor retry are already
        enabled inside ``ActuatorController``) and judge the last read.

        Raises RuntimeError naming each motor that is not at zero so the user knows
        exactly which arm joint to home before re-running.
        """
        assert self._arm is not None, "ActuatorController must be connected"
        qpos = np.zeros(12, dtype=np.float64)
        for _ in range(max(1, settle_reads)):
            qpos = self._read_qpos12_raw()

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
        prev_servo = [0.0] * 12
        accum = [0.0] * 12
        robstride_ref = np.zeros(12, dtype=np.float64)

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            angles = dt.get_joint_angles_from_motors(self._leader_motors)
            a12 = dt.pad12(angles)
            if len(angles) < 12:
                time.sleep(period)
                continue

            if not teleop_initialized:
                if self._arm is not None:
                    # Keep control references in the motor's native encoder frame.
                    # Software-unwrapped qpos is for safety checks / logging only.
                    robstride_ref = self._arm.read_joints12().astype(np.float64)
                prev_servo = list(a12)
                teleop_initialized = True
                # Seed published state with the initial follower pose.
                with self._lock:
                    q0 = self._read_qpos12()
                    self._qpos12[:] = q0.astype(np.float32)
                    self._action12[:] = q0.astype(np.float32)
                continue

            # Accumulate leader-side servo deltas (in raw servo units).
            for i in range(12):
                accum[i] += dt.shortest_delta_units(prev_servo[i], a12[i])
                prev_servo[i] = a12[i]

            # Compute 12-vector follower targets (initial ref + accumulated delta).
            targets = np.zeros(12, dtype=np.float64)
            for i, (_, motor_id) in enumerate(self._left_motors_meta):
                servo_idx = self.LEFT_SERVO_INDICES[i]
                d_rad = dt.accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
                targets[i] = robstride_ref[i] + d_rad
            for i, (_, motor_id) in enumerate(self._right_motors_meta):
                servo_idx = self.RIGHT_SERVO_INDICES[i]
                d_rad = dt.accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
                targets[6 + i] = robstride_ref[6 + i] + d_rad

            # Read qpos *before* commanding so MIT status frames do not corrupt the read.
            if self._arm is not None:
                qpos, raw_encoder = self._read_qpos12_for_command()
                sent = self._arm.command_joints12(targets, ramp=True, feedback12=raw_encoder)
                action = _normalize_qpos_to_pi(sent.astype(np.float64) - self._boot_offsets)
            else:
                sent = targets
                qpos = np.zeros(12, dtype=np.float64)
                action = qpos

            with self._lock:
                self._qpos12[:] = qpos.astype(np.float32)
                self._action12[:] = action.astype(np.float32)

            elapsed = time.monotonic() - t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        with self._lock:
            obs["qpos"] = self._qpos12.copy()
        obs["qvel"] = np.zeros(12, dtype=np.float32)
        obs["effort"] = np.zeros(12, dtype=np.float32)
        return obs

    def get_action(self) -> np.ndarray:
        with self._lock:
            return self._action12.copy()

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


def _start_esc_stop_listener() -> tuple[threading.Event, threading.Event, Callable[[], None]]:
    """
    Background: set `user_stop` when the user presses Esc in the terminal.

    Returns (user_stop, shutdown, join_timeout) where caller must set `shutdown` and call
    `join_timeout()` when recording ends so the thread exits (short join, non-fatal).
    """
    user_stop = threading.Event()
    shutdown = threading.Event()

    def join_timeout() -> None:
        shutdown.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    thread: threading.Thread | None = None

    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return user_stop, shutdown, join_timeout

        def _win_loop() -> None:
            while not shutdown.is_set():
                if msvcrt.kbhit():
                    c = msvcrt.getch()
                    if c == b"\x1b":
                        user_stop.set()
                        return
                time.sleep(0.02)

        thread = threading.Thread(target=_win_loop, daemon=True)
        thread.start()
        return user_stop, shutdown, join_timeout

    # POSIX: raw-ish stdin to detect lone Escape vs arrow sequences
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return user_stop, shutdown, join_timeout

    def _posix_loop() -> None:
        old: list[Any] | None = None
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while not shutdown.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch != "\x1b":
                    continue
                r2, _, _ = select.select([sys.stdin], [], [], 0.04)
                if r2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        r3, _, _ = select.select([sys.stdin], [], [], 0.04)
                        if r3:
                            sys.stdin.read(1)
                    continue
                user_stop.set()
                return
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
    return user_stop, shutdown, join_timeout


def get_next_episode_index(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in output_dir.glob("episode_*.hdf5"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    return 0 if not ids else max(ids) + 1


def record_episode(
    robot: RobotInterface,
    output_path: Path,
    task: str,
    max_steps: int,
    dt: float,
    include_qvel: bool = True,
    include_effort: bool = True,
) -> tuple[bool, bool]:
    robot.on_episode_start()
    obs_qpos, act, obs_qvel, obs_effort = [], [], [], []
    imgs = {cam: [] for cam in CAMERA_NAMES}
    ts = []

    esc_stop, esc_shutdown, esc_join = _start_esc_stop_listener()
    t0 = time.time()
    n = 0
    interrupted = False
    try:
        for _ in range(max_steps):
            if esc_stop.is_set():
                print("\nEnd of episode (Esc). Saving frames captured so far...", flush=True)
                break
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
            n += 1
            time.sleep(max(0.0, dt - (time.time() - step_start)))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped early (Ctrl+C). Saving frames captured so far...", flush=True)
    finally:
        esc_shutdown.set()
        esc_join()
        robot.on_episode_end()

    if n == 0:
        return False, interrupted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as root:
        root.attrs["task"] = task
        root.attrs["fps"] = 1.0 / dt if dt > 0 else 0.0
        root.create_dataset("/timestamp", data=np.asarray(ts, dtype=np.float64))
        obs_grp = root.create_group("observations")
        img_grp = obs_grp.create_group("images")
        for cam in CAMERA_NAMES:
            img_grp.create_dataset(cam, data=np.asarray(imgs[cam], dtype=np.uint8), compression="gzip")
        qpos_arr = np.asarray(obs_qpos, dtype=np.float32)
        qpos_arr = _normalize_qpos_to_pi(qpos_arr).astype(np.float32)
        obs_grp.create_dataset("qpos", data=qpos_arr)
        root.create_dataset("action", data=np.asarray(act, dtype=np.float32))
        if include_qvel and obs_qvel:
            obs_grp.create_dataset("qvel", data=np.asarray(obs_qvel, dtype=np.float32))
        if include_effort and obs_effort:
            obs_grp.create_dataset("effort", data=np.asarray(obs_effort, dtype=np.float32))
    return True, interrupted


def main() -> None:
    p = argparse.ArgumentParser(description="Record 3-camera episodes for GR00T-compatible conversion")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", type=str, default="demo task")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--dt", type=float, default=1.0 / 30.0)
    p.add_argument("--state-dim", type=int, default=12)
    p.add_argument("--action-dim", type=int, default=12)
    p.add_argument("--image-height", type=int, default=640)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--robot", choices=["demo", "usb_cam", "direct_teleop"], default="demo")
    p.add_argument("--video-cam-head", type=int, default=4)
    p.add_argument("--video-cam-left-wrist", type=int, default=0)
    p.add_argument("--video-cam-right-wrist", type=int, default=8)
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
    p.add_argument("--episode-idx", type=int, default=None)
    p.add_argument("--no-qvel", action="store_true")
    p.add_argument("--no-effort", action="store_true")
    p.add_argument("--test-cameras", action="store_true")
    p.add_argument("--test-cameras-max", type=int, default=12)
    args = p.parse_args()

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
        robot = USBVideoRobotInterface(
            cam_head_device=args.video_cam_head,
            cam_left_wrist_device=args.video_cam_left_wrist,
            cam_right_wrist_device=args.video_cam_right_wrist,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            image_shape=(args.image_height, args.image_width),
        )
    else:
        try:
            robot = DirectTeleopRobotInterface(
                cam_head_device=args.video_cam_head,
                cam_left_wrist_device=args.video_cam_left_wrist,
                cam_right_wrist_device=args.video_cam_right_wrist,
                leader_port=args.leader_port,
                leader_baud=args.leader_baud,
                control_rate=args.teleop_rate,
                qpos_median_samples=args.qpos_median_samples,
                qpos_median_gap_s=max(0.0, args.qpos_median_gap_ms / 1000.0),
                dry_run=args.dry_run_teleop,
                image_shape=(args.image_height, args.image_width),
            )
        except RuntimeError as exc:
            # Pre-flight zero-pose check (or other init validation) failed: print and exit
            # cleanly without recording anything.  Resources are already released by
            # DirectTeleopRobotInterface._cleanup_partial_init.
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            sys.exit(1)

    ep_idx = args.episode_idx if args.episode_idx is not None else get_next_episode_index(args.output_dir)
    out = args.output_dir / f"episode_{ep_idx:06d}.hdf5"
    print(f"Recording -> {out}")
    print("Starting in 2 seconds...")
    time.sleep(2)
    print("Start teleoperating now. Press Esc in this terminal to stop and save the episode.", flush=True)

    try:
        ok, stopped_early = record_episode(
            robot,
            out,
            task=args.task,
            max_steps=args.max_steps,
            dt=args.dt,
            include_qvel=not args.no_qvel,
            include_effort=not args.no_effort,
        )
        if ok:
            if stopped_early:
                print(f"Saved partial episode ({out.name}); Ctrl+C ended recording early.", flush=True)
            else:
                print(f"Saved episode ({out.name}).", flush=True)
            next_idx = ep_idx + 1
            next_path = args.output_dir / f"episode_{next_idx:06d}.hdf5"
            print(
                f"Next episode: run the same command again; next file will be {next_path.name}.",
                flush=True,
            )
        else:
            print("No data captured.")
    finally:
        if hasattr(robot, "close"):
            robot.close()


if __name__ == "__main__":
    main()
