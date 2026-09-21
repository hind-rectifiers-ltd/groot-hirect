"""Follower joint layout shared by record / convert / policy client (16-DoF).

Per arm: 7 revolute + 1 gripper. Vector order matches ``move_actuators`` /
``direct_teleop``:

  [0:7]   left arm   — shoulder_pitch..wrist_yaw
  [7]     left gripper
  [8:15]  right arm
  [15]    right gripper

When a source provides fewer values (e.g. old 5-DoF arm groups), missing entries
are filled with **0.0**.
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 16
ARM_REVOLUTE_DOF = 7
LEFT_ARM_SLICE = slice(0, 7)
LEFT_GRIPPER_INDEX = 7
RIGHT_ARM_SLICE = slice(8, 15)
RIGHT_GRIPPER_INDEX = 15
GRIPPER_JOINT_INDICES = (LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX)

# Wrist roll / yaw indices (legacy 5+1 mapping inserted zeros here; 7+1 drives them).
ZERO_FOLLOWER_INDICES = (5, 6, 13, 14)

ACTION_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
EXPECTED_GROUP_DIMS: dict[str, int] = {
    "left_arm": ARM_REVOLUTE_DOF,
    "left_gripper": 1,
    "right_arm": ARM_REVOLUTE_DOF,
    "right_gripper": 1,
}

JOINT_LABELS_16 = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "Lg",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "Rg",
)


def pad_vector(vec: np.ndarray | list | tuple, dim: int = NUM_JOINTS) -> np.ndarray:
    """Pad/truncate to ``dim`` with trailing zeros when short.

    If ``vec`` looks like a legacy flat 12-DoF follower vector (5+1 per arm),
    expand it into 16-DoF by inserting zeros at wrist_roll / wrist_yaw.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 12 and dim == NUM_JOINTS:
        return expand_leader12_style_to_16(v)
    if v.size < dim:
        v = np.pad(v, (0, dim - v.size))
    elif v.size > dim:
        v = v[:dim]
    return v.copy()


def expand_leader12_style_to_16(vec12: np.ndarray | list | tuple) -> np.ndarray:
    """
    Map legacy flat 12-DoF [L0..L4, Lg, R0..R4, Rg] → 16-DoF with wrist zeros.

    Missing wrist_roll / wrist_yaw (indices 5,6 and 13,14) are set to 0.0.
    """
    v = np.asarray(vec12, dtype=np.float64).reshape(-1)
    if v.size < 12:
        v = np.pad(v, (0, 12 - v.size))
    out = np.zeros(NUM_JOINTS, dtype=np.float64)
    out[0:5] = v[0:5]
    out[7] = v[5]
    out[8:13] = v[6:11]
    out[15] = v[11]
    # out[5], out[6], out[13], out[14] stay 0
    return out


def pad_group(arr: np.ndarray, expected_dim: int) -> np.ndarray:
    """
    Pad a policy/state group along the last axis to ``expected_dim`` with zeros.

    Accepts (T, D) or (D,). Truncates if longer than expected.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        if a.size < expected_dim:
            return np.pad(a, (0, expected_dim - a.size)).astype(np.float32)
        return a[:expected_dim].astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D group array, got shape {a.shape}")
    d = a.shape[-1]
    if d < expected_dim:
        return np.pad(a, ((0, 0), (0, expected_dim - d))).astype(np.float32)
    if d > expected_dim:
        return a[:, :expected_dim].astype(np.float32)
    return a.astype(np.float32)


def qpos_to_state_dict(qpos: np.ndarray) -> dict[str, np.ndarray]:
    """16D (or shorter, zero-padded) vector -> GR00T state dict (B=1, T=1, D)."""
    q = pad_vector(qpos, NUM_JOINTS).astype(np.float32)
    return {
        "left_arm": q[LEFT_ARM_SLICE].reshape(1, 1, ARM_REVOLUTE_DOF),
        "left_gripper": q[LEFT_GRIPPER_INDEX : LEFT_GRIPPER_INDEX + 1].reshape(1, 1, 1),
        "right_arm": q[RIGHT_ARM_SLICE].reshape(1, 1, ARM_REVOLUTE_DOF),
        "right_gripper": q[RIGHT_GRIPPER_INDEX : RIGHT_GRIPPER_INDEX + 1].reshape(1, 1, 1),
    }


def actions_to_chunk(action: dict[str, np.ndarray]) -> np.ndarray:
    """Decode policy output to (T, 16); pad short groups with zeros."""
    parts: list[np.ndarray] = []
    horizon: int | None = None
    for key in ACTION_KEYS:
        if key not in action:
            raise KeyError(f"Missing action key {key!r}; got {list(action.keys())}")
        arr = np.asarray(action[key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"action[{key!r}] expected (B,T,D), got {arr.shape}")
        slab = arr[0]
        if horizon is None:
            horizon = int(slab.shape[0])
        elif int(slab.shape[0]) != horizon:
            raise ValueError(f"action horizon mismatch for {key!r}: {slab.shape[0]} vs {horizon}")
        parts.append(pad_group(slab.reshape(horizon, -1), EXPECTED_GROUP_DIMS[key]))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def format_joint_vector(vec: np.ndarray, *, precision: int = 4) -> str:
    v = pad_vector(vec, NUM_JOINTS)
    parts = [f"{JOINT_LABELS_16[i]}={float(v[i]):+.{precision}f}" for i in range(NUM_JOINTS)]
    return "  ".join(parts)
