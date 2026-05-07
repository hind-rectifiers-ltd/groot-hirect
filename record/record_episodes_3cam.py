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

    Records:
      - qpos: follower mechanical positions (12D)
      - action: ramp-limited commanded follower targets (12D)
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

        self._dt = dt
        dt.ensure_import_paths(self._repo_root)

        from dynamixel_easy_sdk import Connector

        try:
            from robstride_dynamics import Motor, ParameterType, RobstrideBus
        except Exception:
            from robstride_dynamics.bus import Motor, RobstrideBus
            from robstride_dynamics.protocol import ParameterType

        self._ParameterType = ParameterType
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

        self._dry_run = dry_run
        self._left_bus = self._right_bus = None
        self._left_motors = [(f"motor_{mid}", mid) for mid in dt.LEFT_ROBSTRIDE_IDS]
        self._right_motors = [(f"motor_{mid}", mid) for mid in dt.RIGHT_ROBSTRIDE_IDS]
        if not dry_run:
            self._left_bus, self._left_motors = dt.init_robstride_bus(
                RobstrideBus,
                Motor,
                ParameterType,
                dt.LEFT_CAN,
                dt.LEFT_ROBSTRIDE_IDS,
            )
            self._right_bus, self._right_motors = dt.init_robstride_bus(
                RobstrideBus,
                Motor,
                ParameterType,
                dt.RIGHT_CAN,
                dt.RIGHT_ROBSTRIDE_IDS,
            )

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

    def _read_follower_qpos12(self) -> np.ndarray:
        out = np.zeros(12, dtype=np.float32)
        for i, (name, _mid) in enumerate(self._left_motors):
            if self._left_bus is None:
                continue
            try:
                out[i] = float(self._left_bus.read(name, self._ParameterType.MECHANICAL_POSITION))
            except Exception:
                pass
        for i, (name, _mid) in enumerate(self._right_motors):
            if self._right_bus is None:
                continue
            try:
                out[6 + i] = float(self._right_bus.read(name, self._ParameterType.MECHANICAL_POSITION))
            except Exception:
                pass
        return out

    def _teleop_loop(self, control_rate: float) -> None:
        dt = self._dt
        period = 1.0 / max(control_rate, 1e-3)
        teleop_initialized = False
        prev_servo = [0.0] * 12
        accum = [0.0] * 12
        robstride_ref = {}
        ramped_cmd = {}
        last_ramp_t = None

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            angles = dt.get_joint_angles_from_motors(self._leader_motors)
            a12 = dt.pad12(angles)
            if len(angles) < 12:
                time.sleep(period)
                continue

            if not teleop_initialized:
                for motor_name, _mid in self._left_motors:
                    if self._left_bus:
                        try:
                            robstride_ref[motor_name] = self._left_bus.read(
                                motor_name, self._ParameterType.MECHANICAL_POSITION
                            )
                        except Exception:
                            robstride_ref[motor_name] = 0.0
                    else:
                        robstride_ref[motor_name] = 0.0
                for motor_name, _mid in self._right_motors:
                    if self._right_bus:
                        try:
                            robstride_ref[motor_name] = self._right_bus.read(
                                motor_name, self._ParameterType.MECHANICAL_POSITION
                            )
                        except Exception:
                            robstride_ref[motor_name] = 0.0
                    else:
                        robstride_ref[motor_name] = 0.0
                prev_servo = list(a12)
                teleop_initialized = True
                ramped_cmd = {name: float(robstride_ref.get(name, 0.0)) for name, _ in self._left_motors + self._right_motors}
                last_ramp_t = time.monotonic()
                continue

            for i in range(12):
                accum[i] += dt.shortest_delta_units(prev_servo[i], a12[i])
                prev_servo[i] = a12[i]

            now = time.monotonic()
            if last_ramp_t is None:
                last_ramp_t = now
            ramp_dt = max(1e-4, min(now - last_ramp_t, dt.ROBSTRIDE_RAMP_DT_MAX_S))
            last_ramp_t = now
            max_step = dt.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S * ramp_dt

            for (motor_name, motor_id), servo_idx in zip(self._left_motors, self.LEFT_SERVO_INDICES):
                base = robstride_ref.get(motor_name, 0.0)
                d_rad = dt.accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
                desired = base + d_rad
                prev_cmd = ramped_cmd.get(motor_name, desired)
                target = dt.ramp_toward(prev_cmd, desired, max_step)
                ramped_cmd[motor_name] = target
                if self._left_bus:
                    try:
                        self._left_bus.write_operation_frame(
                            motor_name,
                            target,
                            dt.MOTOR_KP[motor_id],
                            dt.MOTOR_KD[motor_id],
                            0.0,
                            0.0,
                        )
                    except Exception:
                        pass

            for (motor_name, motor_id), servo_idx in zip(self._right_motors, self.RIGHT_SERVO_INDICES):
                base = robstride_ref.get(motor_name, 0.0)
                d_rad = dt.accum_units_to_target_delta_rad(accum[servo_idx], motor_id, servo_idx)
                desired = base + d_rad
                prev_cmd = ramped_cmd.get(motor_name, desired)
                target = dt.ramp_toward(prev_cmd, desired, max_step)
                ramped_cmd[motor_name] = target
                if self._right_bus:
                    try:
                        self._right_bus.write_operation_frame(
                            motor_name,
                            target,
                            dt.MOTOR_KP[motor_id],
                            dt.MOTOR_KD[motor_id],
                            0.0,
                            0.0,
                        )
                    except Exception:
                        pass

            action = np.zeros(12, dtype=np.float32)
            for i, (name, _mid) in enumerate(self._left_motors):
                action[i] = float(ramped_cmd.get(name, 0.0))
            for i, (name, _mid) in enumerate(self._right_motors):
                action[6 + i] = float(ramped_cmd.get(name, 0.0))
            qpos = self._read_follower_qpos12()

            with self._lock:
                self._action12[:] = action
                self._qpos12[:] = qpos

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
        for bus, motors in [(self._left_bus, self._left_motors), (self._right_bus, self._right_motors)]:
            if bus and motors:
                try:
                    for motor_name, _ in motors:
                        bus.write_operation_frame(motor_name, 0.0, 0.0, 0.0, 0.0, 0.0)
                    time.sleep(0.2)
                    for motor_name, _ in motors:
                        bus.disable(motor_name)
                    bus.disconnect()
                except Exception:
                    pass
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
        obs_grp.create_dataset("qpos", data=np.asarray(obs_qpos, dtype=np.float32))
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
        robot = DirectTeleopRobotInterface(
            cam_head_device=args.video_cam_head,
            cam_left_wrist_device=args.video_cam_left_wrist,
            cam_right_wrist_device=args.video_cam_right_wrist,
            leader_port=args.leader_port,
            leader_baud=args.leader_baud,
            control_rate=args.teleop_rate,
            dry_run=args.dry_run_teleop,
            image_shape=(args.image_height, args.image_width),
        )

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
