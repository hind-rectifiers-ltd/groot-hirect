#!/usr/bin/env python3
"""
Live client for the GR00T PolicyServer (run_gr00t_server.py) with the same 3-camera /
12-DoF layout as record_episodes_3cam.py.

Observation layout matches NEW_EMBODIMENT + record/custom_3cam_config.py:
  video: head, left_wrist, right_wrist  — uint8 (B,T,H,W,C)
  state: left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)
  language: annotation.human.task_description

Example:
  uv run python record/policy_client_3cam.py \\
    --host localhost --port 5555 \\
    --task "pick and place the cube" \\
    --robot demo \\
    --rate-hz 5 --max-steps 100

With USB cameras (indices like record_episodes_3cam):
  uv run python record/policy_client_3cam.py \\
    --robot usb_cam \\
    --video-cam-head 4 --video-cam-left-wrist 0 --video-cam-right-wrist 8

RobStride follower command (same CAN layout as direct_teleop / record direct_teleop mode):
  uv run python record/policy_client_3cam.py \\
    --robot robstride \\
    --video-cam-head 4 --video-cam-left-wrist 0 --video-cam-right-wrist 8 \\
    --task "..." --apply-actions --dry-run-robstride  # omit dry-run on real hardware

Control notes:
  - By default, targets are slewed from the **previous commanded** joint set (teleop-style). That
    smooths policy jitter and avoids wrist oscillation from encoder noise / delay. Before the loop,
    we **seed** that state from real encoders so the first step does not snap from garbage.
  - Use ``--ramp-from-feedback`` to slew from measured ``qpos`` every tick instead (can hunt if
    the policy target changes every step).
  - Use ``--policy-ramp-max-speed`` (rad/s) to limit approach speed; default 2.5 is below teleop's 6.
  - Match ``--task`` to training text (see dataset ``meta/tasks.jsonl``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Repo root (parent of record/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gr00t.policy.server_client import PolicyClient

# Same camera IDs as record_episodes_3cam.py
CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
VIDEO_KEYS = ("head", "left_wrist", "right_wrist")
CAM_TO_VIDEO = dict(zip(CAMERA_NAMES, VIDEO_KEYS, strict=True))

LANGUAGE_KEY = "annotation.human.task_description"


def qpos12_to_state_dict(qpos12: np.ndarray) -> dict[str, np.ndarray]:
    """12D vector -> GR00T state dict with per-group shapes (1, 1, D)."""
    q = np.asarray(qpos12, dtype=np.float32).reshape(-1)
    if q.size < 12:
        q = np.pad(q, (0, 12 - q.size))
    left_arm = q[0:5].reshape(1, 1, 5).astype(np.float32)
    left_gripper = q[5:6].reshape(1, 1, 1).astype(np.float32)
    right_arm = q[6:11].reshape(1, 1, 5).astype(np.float32)
    right_gripper = q[11:12].reshape(1, 1, 1).astype(np.float32)
    return {
        "left_arm": left_arm,
        "left_gripper": left_gripper,
        "right_arm": right_arm,
        "right_gripper": right_gripper,
    }


def images_dict_to_video(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Record-style RGB images -> GR00T video dict (B=1, T=1, H, W, C) uint8."""
    out: dict[str, np.ndarray] = {}
    for cam_name, vid_key in CAM_TO_VIDEO.items():
        img = np.asarray(images[cam_name], dtype=np.uint8)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Image {cam_name} must be (H,W,3) RGB uint8, got {img.shape}")
        out[vid_key] = img[None, None, ...]
    return out


def build_gr00t_observation(
    images: dict[str, np.ndarray],
    qpos12: np.ndarray,
    task: str,
) -> dict[str, Any]:
    return {
        "video": images_dict_to_video(images),
        "state": qpos12_to_state_dict(qpos12),
        "language": {LANGUAGE_KEY: [[task]]},
    }


def actions_to_vector12(action: dict[str, np.ndarray], time_index: int = 0) -> np.ndarray:
    """
    Decode policy output dict (unnormalized joint targets) to a single 12D step.

    Each value is (B, T, D); we take batch 0, timestep ``time_index``.
    """
    parts = []
    for key in ("left_arm", "left_gripper", "right_arm", "right_gripper"):
        if key not in action:
            raise KeyError(f"Missing action key {key!r}; got {list(action.keys())}")
        arr = np.asarray(action[key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"action[{key!r}] expected (B,T,D), got {arr.shape}")
        parts.append(arr[0, time_index].reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


# --- Robot backends (aligned with record_episodes_3cam) ---

def _import_record_interfaces():
    """Lazy import to reuse RobotInterface classes from the recorder."""
    rec = _REPO_ROOT / "record" / "record_episodes_3cam.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("record_episodes_3cam", rec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {rec}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PolicyRobstrideDriver:
    """Minimal follower commander: RobStride buses + ramped MIT targets (no leader teleop)."""

    def __init__(
        self,
        dry_run: bool = False,
        *,
        ramp_max_speed_rad_s: float | None = 2.5,
        ramp_from_feedback: bool = False,
    ):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import direct_teleop as dt

        dt.ensure_import_paths(_REPO_ROOT)

        try:
            from robstride_dynamics import Motor, ParameterType, RobstrideBus
        except Exception:
            from robstride_dynamics.bus import Motor, RobstrideBus
            from robstride_dynamics.protocol import ParameterType

        self._dt = dt
        self._ParameterType = ParameterType
        self._dry_run = dry_run
        self._left_bus = self._right_bus = None
        self._left_motors: list[tuple[str, int]] = []
        self._right_motors: list[tuple[str, int]] = []
        self._ramped: dict[str, float] = {}
        # None -> use direct_teleop.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S (typically 6.0)
        self._ramp_max_speed_rad_s = ramp_max_speed_rad_s
        self._ramp_from_feedback = ramp_from_feedback

        if not dry_run:
            self._left_bus, self._left_motors = dt.init_robstride_bus(
                RobstrideBus, Motor, ParameterType, dt.LEFT_CAN, dt.LEFT_ROBSTRIDE_IDS
            )
            self._right_bus, self._right_motors = dt.init_robstride_bus(
                RobstrideBus, Motor, ParameterType, dt.RIGHT_CAN, dt.RIGHT_ROBSTRIDE_IDS
            )

    def read_qpos12(self) -> np.ndarray:
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

    def sync_ramped_from_feedback(self) -> None:
        """Seed commanded-ramp state from encoders so the first ``command_a12`` starts at true pose."""
        q = self.read_qpos12()
        for idx in range(12):
            if idx < 6:
                motor_name, _ = self._left_motors[idx]
            else:
                motor_name, _ = self._right_motors[idx - 6]
            self._ramped[motor_name] = float(q[idx])

    def command_a12(self, target12: np.ndarray) -> None:
        if self._dry_run or (self._left_bus is None and self._right_bus is None):
            return
        dt = self._dt
        t = np.asarray(target12, dtype=np.float32).reshape(-1)
        if t.size < 12:
            t = np.pad(t, (0, 12 - t.size))
        now = time.monotonic()
        if not hasattr(self, "_last_cmd_t"):
            self._last_cmd_t = now
        ramp_dt = max(1e-4, min(now - self._last_cmd_t, dt.ROBSTRIDE_RAMP_DT_MAX_S))
        self._last_cmd_t = now
        max_speed = (
            dt.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S
            if self._ramp_max_speed_rad_s is None
            else float(self._ramp_max_speed_rad_s)
        )
        max_step = max_speed * ramp_dt

        q_meas = self.read_qpos12()

        for idx in range(12):
            if idx < 6:
                motor_name, mid = self._left_motors[idx]
                bus = self._left_bus
            else:
                motor_name, mid = self._right_motors[idx - 6]
                bus = self._right_bus
            if bus is None:
                continue
            desired = float(t[idx])
            if self._ramp_from_feedback:
                prev = float(q_meas[idx])
            else:
                prev = self._ramped.get(motor_name, float(q_meas[idx]))
            cmd = dt.ramp_toward(prev, desired, max_step)
            self._ramped[motor_name] = cmd
            try:
                bus.write_operation_frame(
                    motor_name,
                    cmd,
                    dt.MOTOR_KP[mid],
                    dt.MOTOR_KD[mid],
                    0.0,
                    0.0,
                )
            except Exception:
                pass

    def close(self) -> None:
        dt = self._dt
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


def main() -> None:
    p = argparse.ArgumentParser(description="GR00T ZMQ policy client (3 cam + 12 DoF)")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--timeout-ms", type=int, default=120000)
    p.add_argument("--task", type=str, default="perform the manipulation task")
    p.add_argument("--rate-hz", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=10**9)
    p.add_argument("--action-time-index", type=int, default=0, help="Which horizon slot to execute (0..15)")
    p.add_argument("--robot", choices=["demo", "usb_cam", "robstride"], default="demo")
    p.add_argument("--video-cam-head", type=int, default=4)
    p.add_argument("--video-cam-left-wrist", type=int, default=0)
    p.add_argument("--video-cam-right-wrist", type=int, default=8)
    p.add_argument("--image-height", type=int, default=640)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--apply-actions", action="store_true", help="Send decoded targets to RobStride (robstride only)")
    p.add_argument("--dry-run-robstride", action="store_true", help="Init driver but do not write CAN")
    p.add_argument(
        "--policy-ramp-max-speed",
        type=float,
        default=2.5,
        help="Max approach speed (rad/s) toward policy target per joint. Lower = slower/smoother. "
        "Use 0 to fall back to direct_teleop default (often 6.0). Default: 2.5",
    )
    p.add_argument(
        "--ramp-from-feedback",
        action="store_true",
        help="Each tick, slew from measured encoder position (can oscillate if policy jitters). "
        "Default: slew from previous command (smoother), after one-time encoder sync at start.",
    )
    args = p.parse_args()

    image_shape = (args.image_height, args.image_width)
    rec = _import_record_interfaces()
    robot: Any = None
    driver: PolicyRobstrideDriver | None = None

    if args.robot == "demo":
        robot = rec.DemoRobotInterface(state_dim=12, action_dim=12, image_shape=image_shape)
    elif args.robot == "usb_cam":
        robot = rec.USBVideoRobotInterface(
            cam_head_device=args.video_cam_head,
            cam_left_wrist_device=args.video_cam_left_wrist,
            cam_right_wrist_device=args.video_cam_right_wrist,
            state_dim=12,
            action_dim=12,
            image_shape=image_shape,
        )
    else:
        robot = rec.USBVideoRobotInterface(
            cam_head_device=args.video_cam_head,
            cam_left_wrist_device=args.video_cam_left_wrist,
            cam_right_wrist_device=args.video_cam_right_wrist,
            state_dim=12,
            action_dim=12,
            image_shape=image_shape,
        )
        ramp_cap = None if args.policy_ramp_max_speed == 0 else args.policy_ramp_max_speed
        driver = PolicyRobstrideDriver(
            dry_run=args.dry_run_robstride,
            ramp_max_speed_rad_s=ramp_cap,
            ramp_from_feedback=args.ramp_from_feedback,
        )

    client = PolicyClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.timeout_ms,
        strict=False,
    )

    if not client.ping():
        print(f"ERROR: server not reachable at tcp://{args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    print(f"Connected to GR00T server tcp://{args.host}:{args.port}")
    if driver is not None and args.robot == "robstride" and args.apply_actions and not args.dry_run_robstride:
        driver.sync_ramped_from_feedback()
        print(
            "RobStride ramp: from_feedback="
            f"{args.ramp_from_feedback}  max_speed_rad_s="
            f"{args.policy_ramp_max_speed if args.policy_ramp_max_speed != 0 else 'direct_teleop default'}"
        )

    period = 1.0 / max(args.rate_hz, 1e-3)
    step = 0
    try:
        while step < args.max_steps:
            t0 = time.monotonic()
            raw = robot.get_observation()
            qpos = np.asarray(raw["qpos"], dtype=np.float32).reshape(-1)
            if driver is not None:
                qpos = driver.read_qpos12()

            obs = build_gr00t_observation(raw["images"], qpos, args.task)
            action, info = client.get_action(obs)
            a12 = actions_to_vector12(action, time_index=args.action_time_index)

            if args.robot == "robstride" and driver is not None and args.apply_actions:
                driver.command_a12(a12)

            print(
                f"step {step}  a12[:4]={np.array2string(a12[:4], precision=3)}  "
                f"qpos[:4]={np.array2string(qpos[:4], precision=3)}"
            )

            step += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if driver is not None:
            driver.close()
        if hasattr(robot, "close"):
            robot.close()


if __name__ == "__main__":
    main()
