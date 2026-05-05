# Recording Toolkit for Custom 3-Camera Humanoid

This folder provides a minimal workflow to collect data for GR00T VLA in the schema we discussed:

- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`
- State/action dims: `12` (left arm 5 + left gripper 1 + right arm 5 + right gripper 1)
- Recording format: per-episode HDF5 (`episode_XXXXXX.hdf5`)
- Conversion output: GR00T-flavored LeRobot v2 dataset with `meta/modality.json`

## Files

- `record_episodes_3cam.py`: records raw HDF5 episodes.
- `convert_3cam_to_groot_lerobot.py`: converts HDF5 episodes to LeRobot v2 and writes `meta/modality.json`.
- `visualize_recorded_episodes.py`: local multi-camera viewer (similar idea to the [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) on Hugging Face).

## Visualize recorded data (local)

Plays back `episode_*.hdf5` with **head | left wrist | right wrist** in one horizontal strip, task text overlay, and **time-series plots** of full-episode **qpos** and **action** (all joints) with a **yellow frame cursor**, plus keyboard controls (like scrubbing a LeRobot-style dataset).

**One session folder** (matches `--session` from recording):

```bash
uv run python record/visualize_recorded_episodes.py --data-dir ./record/cube_pick_place
```

**All episodes under `./record` recursively** (every session):

```bash
uv run python record/visualize_recorded_episodes.py --data-dir ./record --recursive
```

**Export a preview video** (no GUI):

```bash
uv run python record/visualize_recorded_episodes.py \
  --data-dir ./record/cube_pick_place \
  --episode 0 \
  --save-video /tmp/preview_cube.mp4
```

**OpenCV vs headless:** the repo often uses `opencv-python-headless`, which has no `imshow`. The visualizer **auto-switches to a Matplotlib window** in that case. To force OpenCV windows, install `uv pip install opencv-python`, or pass `--gui matplotlib` / `--gui opencv` explicitly. Use `--save-video` for a file-only export with no display.

## 1) Record demos

Example (USB cameras only, zero state/action placeholders):

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir ./record/raw \
  --robot usb_cam \
  --video-cam-head 12 \
  --video-cam-left-wrist 5 \
  --video-cam-right-wrist 1 \
  --task "pick up the object and place it in the tray"
```

If you already have robot state/action from your own stack, plug your backend into `RobotInterface` in the script.

Example (direct teleop + RobStride follower + 3 cameras):

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir ./record/pickplace \
  --robot direct_teleop \
  --leader-port /dev/ttyACM0 \
  --leader-baud 57600 \
  --teleop-rate 10 \
  --video-cam-head 12 \
  --video-cam-left-wrist 5 \
  --video-cam-right-wrist 1 \
  --task "pick up the object and place it in the tray"
```

Find working camera indices first:

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir ./record/raw \
  --test-cameras \
  --test-cameras-max 16
```

### During recording

After a 2-second countdown, the script prints `**Start teleoperating now.**` — that is when recording has begun.

**End an episode early:** press **Ctrl+C**. Whatever frames were captured are saved as a **partial** episode (useful if you finish the task before `--max-steps`).

**Next episode:** run the **same command again** in the same shell (or use the Up arrow to recall it). The script auto-picks the next index (`episode_000000.hdf5`, `episode_000001.hdf5`, …). The final log line tells you the next filename.

**Override index (optional):** pass `--episode-idx N` if you need a specific number.

## 2) Convert to GR00T LeRobot v2

```bash
uv run python record/convert_3cam_to_groot_lerobot.py \
  --raw-dir ./record/raw \
  --repo-id my_robot/pickplace_3cam \
  --fps 30 \
  --state-dim 12 \
  --action-dim 12
```

This writes output under `${HF_LEROBOT_HOME}/my_robot/pickplace_3cam` (default under Hugging Face cache, typically `~/.cache/huggingface/lerobot`). **Current `lerobot` creates LeRobot v3.0 layout** (chunked `data/` parquet, `meta/episodes/` parquet, `meta/tasks.parquet`, …).

GR00T’s data loader expects **LeRobot v2.1-style files** (`meta/episodes.jsonl`, `meta/tasks.jsonl`, **one parquet file per episode**). After conversion, run the repo’s **v3 → v2.1** script once, then restore `meta/modality.json`.

### 2.1) Convert dataset to GR00T-compatible LeRobot v2.1

From the Isaac-GR00T repo root:

```bash
uv run python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id my_robot/pickplace_3cam \
  --root /home/yash/.cache/huggingface/lerobot
```

- `--root` is the **parent directory** of your dataset (usually `$HF_LEROBOT_HOME` / `~/.cache/huggingface/lerobot`).
- The script backs up the v3.0 tree to `pickplace_3cam_v3.0/` (same parent directory as the dataset) and replaces the dataset folder with a v2.1 layout.

The conversion **does not copy** `meta/modality.json`. Restore it from the backup:

```bash
DS=/home/yash/.cache/huggingface/lerobot/my_robot/pickplace_3cam
# Backup folder name is `<dataset_stem>_v3.0`, not `*_v30` (see convert_v3_to_v2.py).
cp "${DS}_v3.0/meta/modality.json" "${DS}/meta/modality.json"
```

Optional (only if you generated `relative_stats.json` before converting):

```bash
test -f "${DS}_v3.0/meta/relative_stats.json" && cp "${DS}_v3.0/meta/relative_stats.json" "${DS}/meta/"
```

Then run **§4 stats** on `"${DS}"` if `meta/stats.json` is missing (the v3→v2 script copies `stats.json` when present).

## 3) Modality config for `NEW_EMBODIMENT` (required)

`NEW_EMBODIMENT` is not built into `MODALITY_CONFIGS` until a Python config **registers** it via `register_modality_config()`.

This repo ships **`record/custom_3cam_config.py`**, which matches the `meta/modality.json` produced by the converter (3 cams, 12-DoF split into four groups, 16-step action horizon, absolute joint actions as a simple baseline). **Import that file** before any tool that needs the tag (stats, finetune, server) by passing:

`--modality-config-path record/custom_3cam_config.py`

To customize (e.g. relative arm deltas, different horizon), copy the file and edit; then point `--modality-config-path` at your copy.

## 4) Generate dataset statistics (required)

After conversion, run (note **`--modality-config-path`** — without it, `new_embodiment` is not registered and stats will fail):

```bash
uv run python gr00t/data/stats.py \
  --dataset-path <converted_dataset_path> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py
```

This generates:

- `meta/stats.json`
- `meta/relative_stats.json` (only for action subspaces marked `RELATIVE` in the modality config; with the shipped file, relative stats are typically empty)

If you change action horizon or modality keys in the config, rerun this command.

## 5) Fine-tune GR00T

Single-GPU example:

```bash
export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 uv run python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <converted_dataset_path> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --num-gpus $NUM_GPUS \
  --output-dir /tmp/gr00t_custom_3cam \
  --max-steps 10000 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --global-batch-size 32 \
  --dataloader-num-workers 4
```

## 6) Open-loop evaluation (sanity check)

Run evaluation on a few trajectories before real-robot deployment:

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path <converted_dataset_path> \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/gr00t_custom_3cam/checkpoint-10000 \
  --traj-ids 0 1 2 \
  --action-horizon 16 \
  --steps 300
```

If predictions diverge heavily from GT, improve data quality/coverage before deployment.

## 7) Deploy on edge (TensorRT path)

### 7.1 Export ONNX from checkpoint

```bash
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path /tmp/gr00t_custom_3cam/checkpoint-10000
```

### 7.2 Build TensorRT pipeline

```bash
uv run python scripts/deployment/build_trt_pipeline.py
```

### 7.3 Benchmark latency on target edge device

```bash
uv run python scripts/deployment/benchmark_inference.py
```

## 8) Real robot inference server/client

Start server (on edge or host with the model):

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/gr00t_custom_3cam/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py
```

Run your robot client loop against the policy server (host/port defaults are `127.0.0.1:5555`).

---

## Suggested execution checklist

1. Record 20-50 demo episodes and verify camera sync/state-action quality.
2. Convert to LeRobot v2 and verify `meta/modality.json`.
3. Generate stats and run a short (1k-2k step) fine-tune smoke test.
4. Run open-loop eval and inspect plots.
5. Scale data and full fine-tune.
6. Export ONNX -> TensorRT -> benchmark on edge.
7. Start server and validate closed-loop behavior on robot.

