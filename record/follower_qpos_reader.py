"""Resilient 16-DoF follower qpos reads (RobStride CAN) for teleop diagnostics.

Prefer ``MECHANICAL_POSITION`` with retries, then MIT ``OPERATION_STATUS`` sweep /
per-motor drain, then last-good per joint (never silent zeros when a prior sample
exists).

Does not modify ``robstride_control``; only calls public ``RobstrideBus`` APIs.

Note: values are in the bus/calibration frame (not ``ActuatorController`` logical
start-pose offsets). For logical-frame qpos use ``ActuatorController.read_joints``.
"""

from __future__ import annotations

import struct
import sys
import time
from typing import Any

import numpy as np

_DEFAULT_CAN_READ_RETRIES = 8
_DEFAULT_CAN_READ_RETRY_DELAY_S = 0.001
_DEFAULT_MIT_SWEEP_TIMEOUT_S = 0.4
_DEFAULT_MIT_SWEEP_MAX_FRAMES = 320

# 7 revolute + gripper per arm (matches move_actuators / direct_teleop).
NUM_JOINTS = 16
ARM_DOF = 8


class ResilientFollowerQposReader:
    def __init__(
        self,
        parameter_type: Any,
        *,
        can_read_retries: int = _DEFAULT_CAN_READ_RETRIES,
        can_read_retry_delay_s: float = _DEFAULT_CAN_READ_RETRY_DELAY_S,
        mit_sweep_timeout_s: float = _DEFAULT_MIT_SWEEP_TIMEOUT_S,
        mit_sweep_max_frames: int = _DEFAULT_MIT_SWEEP_MAX_FRAMES,
        verbose: bool = False,
    ):
        self._ParameterType = parameter_type
        self._can_read_retries = max(1, int(can_read_retries))
        self._can_read_retry_delay_s = max(0.0, float(can_read_retry_delay_s))
        self._mit_sweep_timeout_s = float(mit_sweep_timeout_s)
        self._mit_sweep_max_frames = int(mit_sweep_max_frames)
        self._verbose = verbose
        self._qpos_last_ok = np.full(NUM_JOINTS, np.nan, dtype=np.float32)
        self._can_read_calls = 0
        self._can_read_failures = 0
        self._last_warn_t = 0.0

    def reset_last_ok(self) -> None:
        """Clear last-good cache (e.g. new session where stale hold is undesirable)."""
        self._qpos_last_ok[:] = np.nan

    def stats_line(self) -> str:
        bad_rate = 100.0 * self._can_read_failures / max(self._can_read_calls, 1)
        return (
            f"follower qpos reads: calls={self._can_read_calls} joint-fallbacks="
            f"{self._can_read_failures} ({bad_rate:.1f}% of read_qpos calls had >=1 joint fallback)"
        )

    def _try_read_mechanical_position(self, bus: Any, motor_name: str) -> float:
        last_err: BaseException | None = None
        for attempt in range(self._can_read_retries):
            try:
                return float(bus.read(motor_name, self._ParameterType.MECHANICAL_POSITION))
            except AssertionError as err:
                last_err = err
                if attempt + 1 < self._can_read_retries and self._can_read_retry_delay_s > 0:
                    time.sleep(self._can_read_retry_delay_s)
            except Exception as err:
                last_err = err
                break
        assert last_err is not None
        raise last_err

    def _collect_mit_positions_bus(self, bus: Any, motors: list[tuple[str, int]]) -> list[float]:
        if bus is None or not motors:
            return []

        from robstride_dynamics.protocol import CommunicationType
        from robstride_dynamics.table import MODEL_MIT_POSITION_TABLE

        wanted_names = [name for name, _ in motors]
        target_ids = {bus.motors[name].id: name for name in wanted_names}
        values: dict[str, float] = {}
        deadline = time.monotonic() + self._mit_sweep_timeout_s
        frames_seen = 0

        while len(values) < len(wanted_names) and frames_seen < self._mit_sweep_max_frames:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            received = bus.receive(timeout=max(remaining, 1e-3))
            frames_seen += 1
            if not received:
                continue
            communication_type, extra_data, host_id, data = received
            if communication_type != CommunicationType.OPERATION_STATUS:
                continue
            if len(data) < 8:
                continue
            device_id = (extra_data >> 0) & 0xFF
            motor_key = device_id if device_id in target_ids else host_id
            if motor_key not in target_ids:
                continue
            motor_name = target_ids[motor_key]
            model = bus.motors[motor_name].model
            position_u16, _, _, _ = struct.unpack(">HHHH", data)
            position = (float(position_u16) / 0x7FFF - 1.0) * MODEL_MIT_POSITION_TABLE[model]
            if bus.calibration:
                cal = bus.calibration[motor_name]
                position = (position - cal["homing_offset"]) * cal["direction"]
            values[motor_name] = float(position)

        if len(values) < len(wanted_names):
            missing = [n for n in wanted_names if n not in values]
            raise RuntimeError(
                f"MIT status sweep incomplete: got {len(values)}/{len(wanted_names)} motors; "
                f"missing={missing}"
            )

        return [values[name] for name in wanted_names]

    def _read_single_mit_position(self, bus: Any, motor_name: str) -> float:
        from robstride_dynamics.protocol import CommunicationType
        from robstride_dynamics.table import MODEL_MIT_POSITION_TABLE

        target_mid = bus.motors[motor_name].id
        deadline = time.monotonic() + self._mit_sweep_timeout_s
        frames_seen = 0
        while frames_seen < self._mit_sweep_max_frames:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            received = bus.receive(timeout=max(remaining, 1e-3))
            frames_seen += 1
            if not received:
                continue
            communication_type, extra_data, host_id, data = received
            if communication_type != CommunicationType.OPERATION_STATUS:
                continue
            if len(data) < 8:
                continue
            device_id = (extra_data >> 0) & 0xFF
            motor_key = device_id if device_id == target_mid else host_id
            if motor_key != target_mid:
                continue
            model = bus.motors[motor_name].model
            position_u16, _, _, _ = struct.unpack(">HHHH", data)
            position = (float(position_u16) / 0x7FFF - 1.0) * MODEL_MIT_POSITION_TABLE[model]
            if bus.calibration:
                cal = bus.calibration[motor_name]
                position = (position - cal["homing_offset"]) * cal["direction"]
            return float(position)

        raise RuntimeError(f"No MIT OPERATION_STATUS for {motor_name} (id={target_mid}) in time")

    def _log_failure(self, joint_idx: int, motor_name: str, exc: BaseException, caller: str) -> None:
        if not self._verbose:
            return
        now = time.monotonic()
        if now - self._last_warn_t < 0.25:
            return
        self._last_warn_t = now
        print(
            f"WARN follower_qpos_reader [{caller}]: joint={joint_idx} motor={motor_name!r}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    def read_qpos16(
        self,
        left_bus: Any,
        right_bus: Any,
        left_motors: list[tuple[str, int]],
        right_motors: list[tuple[str, int]],
        *,
        _caller: str = "read_qpos16",
    ) -> np.ndarray:
        """Return 16D qpos; on persistent failure per joint, reuse last good (not 0)."""
        self._can_read_calls += 1
        out = np.zeros(NUM_JOINTS, dtype=np.float32)

        def fill_side(
            bus: Any,
            motors: list[tuple[str, int]],
            base_idx: int,
        ) -> None:
            if bus is None or not motors:
                return
            failed_idx: list[int] = []
            for i, (name, _) in enumerate(motors):
                idx = base_idx + i
                try:
                    v = self._try_read_mechanical_position(bus, name)
                    out[idx] = v
                    self._qpos_last_ok[idx] = v
                except Exception:
                    failed_idx.append(i)

            if not failed_idx:
                return

            try:
                vals = self._collect_mit_positions_bus(bus, motors)
                for i in failed_idx:
                    idx = base_idx + i
                    out[idx] = float(vals[i])
                    self._qpos_last_ok[idx] = float(vals[i])
                return
            except Exception:
                pass

            for i in failed_idx:
                name = motors[i][0]
                idx = base_idx + i
                try:
                    v = self._read_single_mit_position(bus, name)
                    out[idx] = v
                    self._qpos_last_ok[idx] = v
                except Exception as e:
                    self._can_read_failures += 1
                    self._log_failure(idx, name, e, _caller)
                    if not np.isnan(self._qpos_last_ok[idx]):
                        out[idx] = float(self._qpos_last_ok[idx])

        fill_side(left_bus, left_motors, 0)
        fill_side(right_bus, right_motors, ARM_DOF)
        return out

    # Back-compat alias used by direct_teleop / direct_right_hand_teleop.
    read_qpos12 = read_qpos16
