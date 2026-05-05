#!/usr/bin/env python3
"""
Convert raw 3-camera HDF5 episodes into GR00T-flavored LeRobot v2.

Input episode format (from record_episodes_3cam.py):
  /observations/images/{cam_head,cam_left_wrist,cam_right_wrist} : uint8 [T,H,W,3]
  /observations/qpos : float [T,state_dim]
  /observations/qvel : float [T,state_dim] (optional)
  /observations/effort : float [T,state_dim] (optional)
  /action : float [T,action_dim]
  /timestamp : float [T] (optional; falls back to i/fps)
  attr["task"] : str (optional)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    # Older pinned installs (pre ~0.4): lerobot.common.datasets
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[no-redef]
        LEROBOT_HOME as HF_LEROBOT_HOME,
        LeRobotDataset,
    )

    HAVE_LEROBOT_NEW_API = False
else:
    HAVE_LEROBOT_NEW_API = True

CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _load_episode(path: Path):
    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["/observations/qpos"][:], dtype=np.float32)
        action = np.asarray(f["/action"][:], dtype=np.float32)
        qvel = np.asarray(f["/observations/qvel"][:], dtype=np.float32) if "/observations/qvel" in f else None
        effort = (
            np.asarray(f["/observations/effort"][:], dtype=np.float32) if "/observations/effort" in f else None
        )
        ts = np.asarray(f["/timestamp"][:], dtype=np.float64) if "/timestamp" in f else None
        task = f.attrs.get("task", "")
        if isinstance(task, bytes):
            task = task.decode("utf-8")

        imgs = {}
        for cam in CAMERA_NAMES:
            arr = np.asarray(f[f"/observations/images/{cam}"][:], dtype=np.uint8)
            if arr.ndim != 4:
                raise ValueError(f"{path}: camera {cam} has invalid shape {arr.shape}")
            # (T,H,W,C) -> (T,C,H,W)
            if arr.shape[-1] == 3:
                arr = np.transpose(arr, (0, 3, 1, 2))
            imgs[cam] = arr

    n = min(len(qpos), len(action))
    for cam in CAMERA_NAMES:
        n = min(n, len(imgs[cam]))
    if ts is not None:
        n = min(n, len(ts))

    return {
        "qpos": qpos[:n],
        "action": action[:n],
        "qvel": qvel[:n] if qvel is not None else None,
        "effort": effort[:n] if effort is not None else None,
        "timestamp": ts[:n] if ts is not None else None,
        "task": str(task),
        "images": {cam: imgs[cam][:n] for cam in CAMERA_NAMES},
        "length": n,
    }


def _write_modality_json(dataset_path: Path) -> None:
    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 5},
            "left_gripper": {"start": 5, "end": 6},
            "right_arm": {"start": 6, "end": 11},
            "right_gripper": {"start": 11, "end": 12},
        },
        "action": {
            "left_arm": {"start": 0, "end": 5},
            "left_gripper": {"start": 5, "end": 6},
            "right_arm": {"start": 6, "end": 11},
            "right_gripper": {"start": 11, "end": 12},
        },
        "video": {
            "head": {"original_key": "observation.images.cam_head"},
            "left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    out = dataset_path / "meta" / "modality.json"
    out.write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 3-cam HDF5 episodes to GR00T-compatible LeRobot v2")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repo id; output is under HF_LEROBOT_HOME (or ~/.cache/huggingface/lerobot if unset)",
    )
    parser.add_argument("--default-task", type=str, default="demo task")
    parser.add_argument("--robot-type", type=str, default="custom_humanoid")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--state-dim", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--use-videos", action="store_true", help="Store camera streams as mp4s instead of images")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing destination dataset first")
    args = parser.parse_args()

    hdf5_files = sorted(args.raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {args.raw_dir}")

    sample = _load_episode(hdf5_files[0])
    c, h, w = sample["images"]["cam_head"].shape[1:]
    has_qvel = sample["qvel"] is not None
    has_effort = sample["effort"] is not None

    motors_state = [f"joint_{i}" for i in range(args.state_dim)]
    motors_action = [f"joint_{i}" for i in range(args.action_dim)]
    features = {
        "observation.state": {"dtype": "float32", "shape": (args.state_dim,), "names": [motors_state]},
        "action": {"dtype": "float32", "shape": (args.action_dim,), "names": [motors_action]},
    }
    if has_qvel:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (args.state_dim,),
            "names": [motors_state],
        }
    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (args.state_dim,),
            "names": [motors_state],
        }
    mode = "video" if args.use_videos else "image"
    for cam in CAMERA_NAMES:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
        }

    dataset_path = HF_LEROBOT_HOME / args.repo_id
    if dataset_path.exists() and args.overwrite:
        shutil.rmtree(dataset_path)
    if dataset_path.exists() and not args.overwrite:
        raise FileExistsError(f"{dataset_path} exists. Use --overwrite to recreate.")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.use_videos,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    for ep_file in tqdm(hdf5_files, desc="Converting episodes"):
        ep = _load_episode(ep_file)
        if ep["length"] == 0:
            continue
        task = ep["task"].strip() or args.default_task
        for i in range(ep["length"]):
            frame = {
                "observation.state": torch.from_numpy(ep["qpos"][i]),
                "action": torch.from_numpy(ep["action"][i]),
            }
            if ep["qvel"] is not None:
                frame["observation.velocity"] = torch.from_numpy(ep["qvel"][i])
            if ep["effort"] is not None:
                frame["observation.effort"] = torch.from_numpy(ep["effort"][i])
            for cam in CAMERA_NAMES:
                frame[f"observation.images.{cam}"] = ep["images"][cam][i]
            if HAVE_LEROBOT_NEW_API:
                frame["task"] = task
            ds.add_frame(frame)
        if HAVE_LEROBOT_NEW_API:
            ds.save_episode()
        else:
            ds.save_episode(task=task)

    if HAVE_LEROBOT_NEW_API:
        ds.finalize()
    else:
        ds.consolidate()
    _write_modality_json(dataset_path)

    print(f"Saved dataset to: {dataset_path}")
    print(f"Added modality file: {dataset_path / 'meta' / 'modality.json'}")
    print("Next: run gr00t/data/stats.py on this dataset before fine-tuning.")


if __name__ == "__main__":
    main()
