#!/usr/bin/env python3
"""
Live client for the GR00T PolicyServer (run_gr00t_server.py) with the same 3-camera /
16-DoF follower layout as record_episodes_3cam.py / direct_teleop.py.

Observation layout matches NEW_EMBODIMENT + record/custom_3cam_config.py:
  video: head, left_wrist, right_wrist  — uint8 (B,T,H,W,C)
  state: left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)
  language: annotation.human.task_description

Missing / short group dims are zero-padded (e.g. old 5-DoF arm policies).

Example:
  uv run python record/policy_client_3cam.py \\
    --host localhost --port 5555 \\
    --task "pick and place the cube" \\
    --robot demo \\
    --rate-hz 5 --max-steps 100

With USB cameras (stable USB ports via record/camera_ports.json on Linux):
  uv run python record/policy_client_3cam.py \\
    --robot usb_cam \\
    --list-cameras-working   # optional: verify ports

RobStride follower command (same CAN layout as direct_teleop / record direct_teleop mode):
  uv run python record/policy_client_3cam.py \\
    --robot robstride \\
    --use-usb-camera-ports \\
    --task "..." --apply-actions --dry-run-robstride  # omit dry-run on real hardware

Control notes:
  - Default ``--control-mode chunk`` runs the full 16-step action horizon before re-inferring (much
    smoother than querying the policy every tick with ``--action-time-index 0`` only).
  - ``--action-smoothing-alpha`` / ``--gripper-smoothing-alpha`` low-pass filter policy targets;
    grippers default to heavier smoothing so jaws can close without jerking open.
  - Targets are slewed from the **previous commanded** joint set (teleop-style). Before the loop,
    ramp state is **seeded** from encoders so the first command does not snap.
  - Use ``--policy-ramp-max-speed`` (rad/s) for MIT slew; ``--max-target-step-gripper`` caps how
    fast the *filtered* gripper target can change per control tick.
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
if str(_REPO_ROOT / "record") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "record"))

from gr00t.policy.server_client import PolicyClient
from joint_layout import (  # noqa: E402
    GRIPPER_JOINT_INDICES,
    JOINT_LABELS_16,
    NUM_JOINTS,
    actions_to_chunk,
    format_joint_vector,
    pad_vector,
    qpos_to_state_dict,
)

# Same camera IDs as record_episodes_3cam.py
CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
VIDEO_KEYS = ("head", "left_wrist", "right_wrist")
CAM_TO_VIDEO = dict(zip(CAMERA_NAMES, VIDEO_KEYS, strict=True))

LANGUAGE_KEY = "annotation.human.task_description"
DEFAULT_ACTION_HORIZON = 16

# Back-compat aliases
JOINT_LABELS_12 = JOINT_LABELS_16
format_joint_vector12 = format_joint_vector
qpos12_to_state_dict = qpos_to_state_dict
actions_to_chunk12 = actions_to_chunk


def log_step_state(
    *,
    step: int,
    qpos: np.ndarray,
    cmd: np.ndarray,
    precision: int = 4,
    extra: str = "",
) -> None:
    """Print full qpos and cmd for all joints."""
    suffix = f"  {extra}" if extra else ""
    print(f"step {step}{suffix}", flush=True)
    print(f"  qpos  {format_joint_vector(qpos, precision=precision)}", flush=True)
    print(f"  cmd   {format_joint_vector(cmd, precision=precision)}", flush=True)


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
    qpos: np.ndarray,
    task: str,
) -> dict[str, Any]:
    return {
        "video": images_dict_to_video(images),
        "state": qpos_to_state_dict(qpos),
        "language": {LANGUAGE_KEY: [[task]]},
    }


def actions_to_vector12(action: dict[str, np.ndarray], time_index: int = 0) -> np.ndarray:
    """Decode policy output to a single 16D step (name kept for back-compat)."""
    chunk = actions_to_chunk(action)
    if time_index < 0 or time_index >= chunk.shape[0]:
        raise IndexError(f"action time_index {time_index} out of range for horizon {chunk.shape[0]}")
    return chunk[time_index].astype(np.float32)


actions_to_vector16 = actions_to_vector12


def upsample_chunk_linear(chunk: np.ndarray, factor: int) -> np.ndarray:
    """Linearly upsample (T, D) chunk to finer control ticks."""
    if factor <= 1:
        return np.asarray(chunk, dtype=np.float32)
    c = np.asarray(chunk, dtype=np.float64)
    if c.shape[0] < 2:
        return np.repeat(c, factor, axis=0).astype(np.float32)
    t_old = np.arange(c.shape[0], dtype=np.float64)
    t_new = np.linspace(0.0, float(c.shape[0] - 1), (c.shape[0] - 1) * factor + 1)
    out = np.zeros((t_new.size, c.shape[1]), dtype=np.float64)
    for j in range(c.shape[1]):
        out[:, j] = np.interp(t_new, t_old, c[:, j])
    return out.astype(np.float32)


class ActionTargetSmoother:
    """Exponential smoothing on policy targets; grippers can use a separate (stronger) alpha."""

    def __init__(
        self,
        *,
        arm_alpha: float = 0.4,
        gripper_alpha: float = 0.65,
        max_arm_step: float = 0.0,
        max_gripper_step: float = 0.025,
        state_dim: int = NUM_JOINTS,
    ):
        self._arm_alpha = float(np.clip(arm_alpha, 0.0, 1.0))
        self._gripper_alpha = float(np.clip(gripper_alpha, 0.0, 1.0))
        self._max_arm_step = float(max(0.0, max_arm_step))
        self._max_gripper_step = float(max(0.0, max_gripper_step))
        self._state_dim = int(state_dim)
        self._state: np.ndarray | None = None

    @property
    def last_output(self) -> np.ndarray | None:
        if self._state is None:
            return None
        return self._state.astype(np.float32)

    def reset(self, qpos: np.ndarray) -> None:
        self._state = pad_vector(qpos, self._state_dim)

    @property
    def enabled(self) -> bool:
        return self._arm_alpha > 0.0 or self._gripper_alpha > 0.0

    def apply(self, target: np.ndarray) -> np.ndarray:
        t = pad_vector(target, self._state_dim)
        if self._state is None:
            self.reset(t)
            return t.astype(np.float32)
        out = self._state.copy()
        for i in range(self._state_dim):
            alpha = self._gripper_alpha if i in GRIPPER_JOINT_INDICES else self._arm_alpha
            if alpha <= 0.0:
                out[i] = float(t[i])
            else:
                out[i] = alpha * out[i] + (1.0 - alpha) * float(t[i])
            max_step = self._max_gripper_step if i in GRIPPER_JOINT_INDICES else self._max_arm_step
            if max_step > 0.0:
                delta = float(np.clip(out[i] - self._state[i], -max_step, max_step))
                out[i] = self._state[i] + delta
        self._state = out
        return out.astype(np.float32)


def blend_chunk_start(chunk: np.ndarray, from_pose: np.ndarray, blend_steps: int) -> np.ndarray:
    """Ease the first ``blend_steps`` rows from ``from_pose`` into the chunk (reduces replan jumps)."""
    if blend_steps <= 0:
        return chunk
    c = np.asarray(chunk, dtype=np.float64).copy()
    start = pad_vector(from_pose, c.shape[1] if c.ndim == 2 else NUM_JOINTS)
    if start.size != c.shape[1]:
        start = pad_vector(start, c.shape[1])
    n = min(blend_steps, c.shape[0])
    for i in range(n):
        w = (i + 1) / float(n)
        c[i] = (1.0 - w) * start + w * c[i]
    return c.astype(np.float32)


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


def _import_usb_cameras():
    """Lazy import shared USB camera port pinning (record/usb_cameras.py)."""
    import importlib.util

    path = _REPO_ROOT / "record" / "usb_cameras.py"
    spec = importlib.util.spec_from_file_location("usb_cameras", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PolicyRobstrideDriver:
    """Minimal follower commander: ramped MIT targets via ``ActuatorController`` (no leader teleop).

    Adds two safety features for inference:
      * **Software auto-zero**: at connect, any joint reading > +π or < -π (i.e. the multi-turn
        encoder reported one wrap away from home) gets a one-turn offset.  All subsequent reads
        return the unwrapped value, and all commands have the offset added back before being
        sent to the motor.  This makes the policy see a clean ``≈0 rad`` home pose even after
        a power cycle wraps the encoder.
      * **Zero-pose check** (:meth:`verify_zero_pose`): refuse to start inference unless every
        joint (after software unwrap) sits within ``[-0.2, +0.2] rad``.
    """

    def __init__(
        self,
        dry_run: bool = False,
        *,
        ramp_max_speed_rad_s: float | None = 2.5,
        ramp_from_feedback: bool = False,
        auto_zero: bool = True,
        safety_max_delta_rad: float = 1.5,
        safety_max_initial_delta_rad: float = 1.0,
        safety_clamp: bool = True,
    ):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import direct_teleop as dt

        dt.ensure_import_paths(_REPO_ROOT)
        from move_actuators import (
            LEFT_ROBSTRIDE_IDS,
            RIGHT_ROBSTRIDE_IDS,
            ActuatorController,
            NUM_JOINTS as ACTUATOR_NUM_JOINTS,
        )

        if int(ACTUATOR_NUM_JOINTS) != NUM_JOINTS:
            raise RuntimeError(
                f"move_actuators NUM_JOINTS={ACTUATOR_NUM_JOINTS} != layout {NUM_JOINTS}"
            )

        self._dt = dt
        self._dry_run = dry_run
        self._ramp_from_feedback = ramp_from_feedback
        self._ramp_max_speed_rad_s = ramp_max_speed_rad_s
        self._motor_ids: list[int] = list(LEFT_ROBSTRIDE_IDS) + list(RIGHT_ROBSTRIDE_IDS)
        # One-turn-unwrap offsets in the motor's encoder frame: read_qpos returns
        # (raw - boot_offsets); command_a sends (target + boot_offsets).
        self._boot_offsets: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._last_raw_encoder: np.ndarray | None = None
        self._arm: ActuatorController | None = None

        if not dry_run:
            ramp_speed = (
                float(ramp_max_speed_rad_s)
                if ramp_max_speed_rad_s is not None
                else float(dt.ROBSTRIDE_RAMP_MAX_SPEED_RAD_S)
            )
            self._arm = ActuatorController(
                ramp=True,
                ramp_max_speed_rad_s=ramp_speed,
                ramp_dt_max_s=float(dt.ROBSTRIDE_RAMP_DT_MAX_S),
                safety_max_delta_rad=float(safety_max_delta_rad),
                safety_max_initial_delta_rad=float(safety_max_initial_delta_rad),
                safety_clamp=bool(safety_clamp),
                safety_abort_on_breach=not bool(safety_clamp),
                read_max_retries=4,
                parallel_bus_reads=True,
            )
            try:
                self._arm.connect()
                if auto_zero:
                    self._capture_boot_offsets()
            except Exception:
                # Make sure we release CAN buses if anything blew up mid-init
                try:
                    self._arm.disconnect(send_zero=False)
                except Exception:
                    pass
                self._arm = None
                raise

    # -- internal helpers -------------------------------------------------

    def _capture_boot_offsets(self, *, settle_reads: int = 3) -> None:
        """Detect one-turn encoder ambiguity and store per-joint offsets.

        On power-up the RobStride multi-turn encoder can land ±2π away from the
        true home position even when the joint is physically at zero.  We
        compensate in software: any reading whose magnitude exceeds π is treated
        as one wrap away.  Joints already inside (-π, +π) get a zero offset.
        """
        assert self._arm is not None
        raw = np.zeros(NUM_JOINTS, dtype=np.float64)
        for _ in range(max(1, settle_reads)):
            raw = self._arm.read_joints().astype(np.float64)
        adjusted: list[tuple[int, float, float]] = []
        for i in range(NUM_JOINTS):
            v = float(raw[i])
            if v > np.pi:
                self._boot_offsets[i] = 2.0 * np.pi
            elif v < -np.pi:
                self._boot_offsets[i] = -2.0 * np.pi
            else:
                self._boot_offsets[i] = 0.0
            if abs(self._boot_offsets[i]) > 0:
                adjusted.append((self._motor_ids[i], v, v - self._boot_offsets[i]))
        if adjusted:
            print("[policy_client] Software zero: one-turn unwrap applied to:", flush=True)
            for mid, before, after in adjusted:
                print(
                    f"  motor id {mid}: encoder={before:+.4f} rad → reported as {after:+.4f} rad",
                    flush=True,
                )
        else:
            print("[policy_client] Software zero: all joints already inside (-π, +π).", flush=True)

    # -- public API used by main() ---------------------------------------

    def _read_raw_encoder_median(self, *, samples: int = 2) -> np.ndarray:
        """Median of ``samples`` raw encoder reads (rejects single garbage CAN frames)."""
        assert self._arm is not None
        n = max(1, int(samples))
        stack = [self._arm.read_joints().astype(np.float64) for _ in range(n)]
        return stack[0] if n == 1 else np.median(np.stack(stack, axis=0), axis=0)

    def read_qpos16(self) -> np.ndarray:
        """Return the 16-DoF follower pose in the *unwrapped* (software-zero) frame."""
        if self._arm is None:
            return np.zeros(NUM_JOINTS, dtype=np.float32)
        raw = self._read_raw_encoder_median()
        self._last_raw_encoder = raw
        return ((raw - self._boot_offsets + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    # Back-compat alias
    read_qpos12 = read_qpos16

    def verify_zero_pose(
        self,
        *,
        low: float = -0.2,
        high: float = 0.2,
        settle_reads: int = 3,
    ) -> None:
        """Refuse to proceed unless every follower joint is at home (~0 rad).

        Mirrors the check in ``record/record_episodes_3cam.py`` but with the
        wider ``[-0.2, +0.2] rad`` tolerance the user requested for inference.
        Raises:
            RuntimeError: with a per-motor breakdown if any joint is outside the
            allowed window.
        """
        if self._arm is None or self._dry_run:
            return
        qpos = np.zeros(NUM_JOINTS, dtype=np.float32)
        for _ in range(max(1, settle_reads)):
            qpos = self.read_qpos16()
        bad: list[tuple[int, float]] = []
        for i in range(NUM_JOINTS):
            v = float(qpos[i])
            if not (low <= v <= high):
                bad.append((self._motor_ids[i], v))
        if bad:
            details = "\n".join(
                f"  motor id {mid} is not zero (qpos={v:+.4f} rad, allowed [{low:+.2f}, {high:+.2f}])"
                for mid, v in bad
            )
            raise RuntimeError(
                "Follower pre-inference zero-pose check failed:\n"
                + details
                + "\nMove the follower arm(s) back to home position (~0 rad) and re-run."
            )
        qpos_str = ", ".join(f"{float(v):+.4f}" for v in qpos)
        print(f"[policy_client] Follower zero-pose check OK: qpos=[{qpos_str}]", flush=True)

    def sync_ramped_from_feedback(self) -> None:
        """Seed the ramp state from current encoders so the first command does not snap."""
        if self._arm is None:
            return
        raw_qpos = self._read_raw_encoder_median(samples=3)
        self._arm.seed_ramp_from_angles(raw_qpos)

    def command_a16(self, target: np.ndarray) -> None:
        if self._arm is None or self._dry_run:
            return
        t = pad_vector(target, NUM_JOINTS)
        if self._ramp_from_feedback:
            self._arm.seed_ramp_from_angles(self._read_raw_encoder_median())
            feedback = None
        else:
            feedback = self._last_raw_encoder
        encoder_target = t + self._boot_offsets
        self._arm.command_joints(encoder_target, ramp=True, feedback12=feedback)

    # Back-compat alias
    command_a12 = command_a16

    def close(self) -> None:
        if self._arm is None:
            return
        try:
            print(self._arm.read_stats_line(), flush=True)
        except Exception:
            pass
        try:
            self._arm.disconnect(send_zero=True)
        except Exception:
            pass
        self._arm = None


def main() -> None:
    p = argparse.ArgumentParser(description="GR00T ZMQ policy client (3 cam + 16 DoF follower)")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--timeout-ms", type=int, default=120000)
    p.add_argument("--task", type=str, default="perform the manipulation task")
    p.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Control loop rate (Hz). Match training teleop (--teleop-rate 10) for chunk mode.",
    )
    p.add_argument("--max-steps", type=int, default=10**9)
    p.add_argument(
        "--control-mode",
        choices=("chunk", "legacy"),
        default="chunk",
        help="chunk: execute full policy horizon before re-inferring (recommended). "
        "legacy: one policy query per tick using --action-time-index only.",
    )
    p.add_argument(
        "--infer-stride",
        type=int,
        default=0,
        help="In chunk mode, re-infer after this many executed steps (0 = full horizon). "
        "Use 8-12 for more reactive control at the cost of some smoothness.",
    )
    p.add_argument(
        "--chunk-upsample",
        type=int,
        default=1,
        help="Linearly upsample each action chunk by this factor before execution (finer motion).",
    )
    p.add_argument(
        "--chunk-blend-steps",
        type=int,
        default=4,
        help="When a new chunk arrives, blend its first N steps from the last smoothed command.",
    )
    p.add_argument(
        "--action-smoothing-alpha",
        type=float,
        default=0.4,
        help="EMA on arm joints: y = alpha*prev + (1-alpha)*target. 0 disables arm smoothing.",
    )
    p.add_argument(
        "--gripper-smoothing-alpha",
        type=float,
        default=0.7,
        help=f"EMA on gripper joints (indices {GRIPPER_JOINT_INDICES}). Higher = smoother/slower jaw motion.",
    )
    p.add_argument(
        "--max-target-step-gripper",
        type=float,
        default=0.025,
        help="Max change (rad) of smoothed gripper target per control tick. 0 = no cap.",
    )
    p.add_argument(
        "--max-target-step-arm",
        type=float,
        default=0.0,
        help="Max change (rad) of smoothed arm target per control tick. 0 = no cap (ramp only).",
    )
    p.add_argument("--action-time-index", type=int, default=0, help="legacy mode: horizon slot (0..15)")
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print full 16-joint qpos/cmd every N control steps (1 = every step).",
    )
    p.add_argument(
        "--log-joint-precision",
        type=int,
        default=4,
        help="Decimal places for per-joint qpos/cmd log lines.",
    )
    p.add_argument(
        "--compact-log",
        action="store_true",
        help="Legacy one-line logs (first 4 joints + grippers only).",
    )
    p.add_argument("--robot", choices=["demo", "usb_cam", "robstride"], default="demo")
    _usb_cameras = _import_usb_cameras()
    _usb_cameras.add_three_camera_cli_args(p)
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
        "--safety-max-delta-rad",
        type=float,
        default=1.5,
        help="Per-tick |ramped target - encoder| safety cap (rad). Default: 1.5",
    )
    p.add_argument(
        "--safety-max-initial-delta-rad",
        type=float,
        default=1.0,
        help="Safety cap on the first command after connect (rad). Default: 1.0",
    )
    p.add_argument(
        "--safety-abort",
        action="store_true",
        help="Abort+disconnect on safety breach instead of clamping (default: clamp).",
    )
    p.add_argument(
        "--ramp-from-feedback",
        action="store_true",
        help="Each tick, slew from measured encoder position (can oscillate if policy jitters). "
        "Default: slew from previous command (smoother), after one-time encoder sync at start.",
    )
    p.add_argument(
        "--no-software-zero",
        action="store_true",
        help="Disable the one-turn-unwrap software zero applied to RobStride encoders at start. "
        "Use only if you have already re-zeroed the motors with vendor tools.",
    )
    p.add_argument(
        "--zero-check-low",
        type=float,
        default=-0.2,
        help="Lower bound (rad) for the pre-inference follower zero-pose check. Default: -0.2",
    )
    p.add_argument(
        "--zero-check-high",
        type=float,
        default=0.2,
        help="Upper bound (rad) for the pre-inference follower zero-pose check. Default: +0.2",
    )
    args = p.parse_args()
    if _usb_cameras.handle_camera_list_flags(args):
        return

    ramp_for_defaults = 6.0 if args.policy_ramp_max_speed == 0 else float(args.policy_ramp_max_speed)
    image_shape = (args.image_height, args.image_width)
    rec = _import_record_interfaces()
    robot: Any = None
    driver: PolicyRobstrideDriver | None = None

    if args.robot == "demo":
        robot = rec.DemoRobotInterface(state_dim=NUM_JOINTS, action_dim=NUM_JOINTS, image_shape=image_shape)
    elif args.robot == "usb_cam":
        camera_devices = _usb_cameras.camera_cli_from_args(args)
        robot = rec.USBVideoRobotInterface(
            camera_devices=camera_devices,
            state_dim=NUM_JOINTS,
            action_dim=NUM_JOINTS,
            image_shape=image_shape,
        )
    else:
        camera_devices = _usb_cameras.camera_cli_from_args(args)
        robot = rec.USBVideoRobotInterface(
            camera_devices=camera_devices,
            state_dim=NUM_JOINTS,
            action_dim=NUM_JOINTS,
            image_shape=image_shape,
        )
        ramp_cap = None if args.policy_ramp_max_speed == 0 else args.policy_ramp_max_speed
        try:
            driver = PolicyRobstrideDriver(
                dry_run=args.dry_run_robstride,
                ramp_max_speed_rad_s=ramp_cap,
                ramp_from_feedback=args.ramp_from_feedback,
                auto_zero=not args.no_software_zero,
                safety_max_delta_rad=args.safety_max_delta_rad,
                safety_max_initial_delta_rad=args.safety_max_initial_delta_rad,
                safety_clamp=not args.safety_abort,
            )
        except Exception as exc:
            print(f"\nERROR: failed to initialize RobStride driver: {exc}", file=sys.stderr, flush=True)
            if hasattr(robot, "close"):
                try:
                    robot.close()
                except Exception:
                    pass
            sys.exit(1)

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

    # Pre-inference safety: refuse to run unless the follower is at home pose.
    # Always run when motors are physically connected; --apply-actions only gates writes.
    if driver is not None and args.robot == "robstride" and not args.dry_run_robstride:
        try:
            driver.verify_zero_pose(low=args.zero_check_low, high=args.zero_check_high)
        except RuntimeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            try:
                driver.close()
            finally:
                if hasattr(robot, "close"):
                    try:
                        robot.close()
                    except Exception:
                        pass
            sys.exit(1)

    smoother: ActionTargetSmoother | None = None
    if args.action_smoothing_alpha > 0.0 or args.gripper_smoothing_alpha > 0.0:
        smoother = ActionTargetSmoother(
            arm_alpha=args.action_smoothing_alpha,
            gripper_alpha=args.gripper_smoothing_alpha,
            max_arm_step=args.max_target_step_arm,
            max_gripper_step=args.max_target_step_gripper,
        )

    if driver is not None and args.robot == "robstride" and args.apply_actions and not args.dry_run_robstride:
        driver.sync_ramped_from_feedback()
        q0 = driver.read_qpos16()
        if smoother is not None:
            smoother.reset(q0)
        print(
            "RobStride ramp: from_feedback="
            f"{args.ramp_from_feedback}  max_speed_rad_s="
            f"{args.policy_ramp_max_speed if args.policy_ramp_max_speed != 0 else 'direct_teleop default'}"
            f"  control_mode={args.control_mode}  rate_hz={args.rate_hz}",
            flush=True,
        )
        if smoother is not None and smoother.enabled:
            print(
                f"  smoothing: arm_alpha={args.action_smoothing_alpha} "
                f"gripper_alpha={args.gripper_smoothing_alpha} "
                f"max_grip_step={args.max_target_step_gripper}",
                flush=True,
            )

    period = 1.0 / max(args.rate_hz, 1e-3)
    step = 0
    chunk_queue: list[np.ndarray] = []
    chunk_plan_id = 0
    last_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)

    def fetch_observation() -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
        raw = robot.get_observation()
        qpos = pad_vector(raw["qpos"], NUM_JOINTS).astype(np.float32)
        if driver is not None:
            qpos = driver.read_qpos16()
        obs = build_gr00t_observation(raw["images"], qpos, args.task)
        return obs, qpos, raw["images"]

    def plan_chunk(obs: dict[str, Any], qpos: np.ndarray) -> np.ndarray:
        nonlocal chunk_plan_id, last_cmd
        action, _info = client.get_action(obs)
        chunk = actions_to_chunk(action)
        if args.chunk_upsample > 1:
            chunk = upsample_chunk_linear(chunk, args.chunk_upsample)
        stride = args.infer_stride if args.infer_stride > 0 else chunk.shape[0]
        stride = min(stride, chunk.shape[0])
        chunk = chunk[:stride]
        if args.chunk_blend_steps > 0:
            chunk = blend_chunk_start(chunk, last_cmd, args.chunk_blend_steps)
        chunk_plan_id += 1
        prec = max(0, args.log_joint_precision)
        print(
            f"[policy_client] plan #{chunk_plan_id}  horizon={chunk.shape[0]}  "
            f"queue will have {chunk.shape[0]} steps",
            flush=True,
        )
        print(f"  qpos  {format_joint_vector(qpos, precision=prec)}", flush=True)
        print(f"  target[0]  {format_joint_vector(chunk[0], precision=prec)}", flush=True)
        return chunk

    def execute_target(raw_target: np.ndarray) -> np.ndarray:
        nonlocal last_cmd
        cmd = smoother.apply(raw_target) if smoother is not None else pad_vector(raw_target, NUM_JOINTS).astype(np.float32)
        last_cmd = cmd.copy()
        if args.robot == "robstride" and driver is not None and args.apply_actions:
            driver.command_a16(cmd)
        return cmd

    def hold_last_command(reason: str, qpos: np.ndarray) -> None:
        """Send the previous valid command when policy output is unavailable/invalid."""
        nonlocal step
        cmd = execute_target(last_cmd)
        print(f"[policy_client] WARN: {reason}; holding last command.", flush=True)
        log_step_state(
            step=step,
            qpos=qpos,
            cmd=cmd,
            precision=args.log_joint_precision,
            extra="HOLD",
        )
        step += 1

    def maybe_log_step(*, qpos: np.ndarray, cmd: np.ndarray, extra: str = "") -> None:
        if args.log_every <= 0 or (step % args.log_every) != 0:
            return
        if args.compact_log:
            g_l, g_r = GRIPPER_JOINT_INDICES
            print(
                f"step {step}  cmd[:4]={np.array2string(cmd[:4], precision=3)}  "
                f"qpos[:4]={np.array2string(qpos[:4], precision=3)}  "
                f"grip={cmd[g_l]:.3f},{cmd[g_r]:.3f}  {extra}",
                flush=True,
            )
        else:
            log_step_state(
                step=step,
                qpos=qpos,
                cmd=cmd,
                precision=args.log_joint_precision,
                extra=extra,
            )

    try:
        while step < args.max_steps:
            t0 = time.monotonic()

            if not chunk_queue:
                obs, qpos, _images = fetch_observation()
                if args.control_mode == "legacy":
                    try:
                        action, _info = client.get_action(obs)
                        raw = actions_to_vector16(action, time_index=args.action_time_index)
                        if not np.all(np.isfinite(raw)):
                            raise ValueError("policy action contains non-finite values")
                        cmd = execute_target(raw)
                        maybe_log_step(qpos=qpos, cmd=cmd, extra="legacy")
                        step += 1
                    except Exception as exc:
                        hold_last_command(f"policy read failed ({exc})", qpos)
                else:
                    try:
                        chunk_queue = list(plan_chunk(obs, qpos))
                    except Exception as exc:
                        hold_last_command(f"policy replan failed ({exc})", qpos)
                        chunk_queue = []

            if chunk_queue:
                raw_target = chunk_queue.pop(0)
                qpos = (
                    driver.read_qpos16()
                    if driver is not None
                    else np.zeros(NUM_JOINTS, dtype=np.float32)
                )
                cmd = execute_target(raw_target)
                maybe_log_step(qpos=qpos, cmd=cmd, extra=f"queue={len(chunk_queue)}")
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