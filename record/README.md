# Recording Toolkit for Custom 3-Camera Humanoid

This folder provides a minimal workflow to collect data for GR00T VLA in the schema we discussed:

- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`
- State/action dims: `12` (left arm 5 + left gripper 1 + right arm 5 + right gripper 1)
- Recording format: per-episode HDF5 (`episode_XXXXXX.hdf5`)
- Conversion output: GR00T-flavored LeRobot v2 dataset with `meta/modality.json`

## Files

- `record_episodes_3cam.py`: records raw HDF5 episodes.
- `convert_3cam_to_groot_lerobot.py`: converts HDF5 episodes to LeRobot v2 and writes `meta/modality.json`.

## 1) Record demos

Example (USB cameras only, zero state/action placeholders):

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir ./record/raw \
  --robot usb_cam \
  --video-cam-head 4 \
  --video-cam-left-wrist 0 \
  --video-cam-right-wrist 8 \
  --task "pick up the object and place it in the tray"
```

If you already have robot state/action from your own stack, plug your backend into `RobotInterface` in the script.

Example (direct teleop + RobStride follower + 3 cameras):

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir ./record/raw \
  --robot direct_teleop \
  --leader-port /dev/ttyACM0 \
  --leader-baud 57600 \
  --teleop-rate 10 \
  --video-cam-head 4 \
  --video-cam-left-wrist 0 \
  --video-cam-right-wrist 8 \
  --task "pick up the object and place it in the tray"
```

Find working camera indices first:

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir ./record/raw \
  --test-cameras \
  --test-cameras-max 16
```

## 2) Convert to GR00T LeRobot v2

```bash
uv run python record/convert_3cam_to_groot_lerobot.py \
  --raw-dir ./record/raw \
  --repo-id my_robot/pickplace_3cam \
  --fps 30 \
  --state-dim 12 \
  --action-dim 12
```

This writes output under `${LEROBOT_HOME}/my_robot/pickplace_3cam` and creates:

- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/modality.json`
- `data/chunk-*/episode_*.parquet`
- `videos/chunk-*/observation.images.*/episode_*.mp4` (when `--use-videos`)

## 3) Generate dataset statistics (required)

After conversion, run:

```bash
uv run python gr00t/data/stats.py \
  --dataset-path <converted_dataset_path> \
  --embodiment-tag NEW_EMBODIMENT
```

This generates:

- `meta/stats.json`
- `meta/relative_stats.json`

## 4) Create your modality config for `NEW_EMBODIMENT`

Create a python config file, for example:

- `record/custom_3cam_config.py`

Use these keys to match `meta/modality.json` from this toolkit:

- video: `head`, `left_wrist`, `right_wrist`
- state: `left_arm`, `left_gripper`, `right_arm`, `right_gripper`
- action: `left_arm`, `left_gripper`, `right_arm`, `right_gripper`
- language: `annotation.human.task_description`

Recommended starting setup:

- video/state delta indices: `[0]`
- action horizon: `list(range(0, 16))`
- arm actions: start with `ABSOLUTE` (simple baseline)
- gripper actions: `ABSOLUTE`

If you later change action horizon or modality keys, rerun `gr00t/data/stats.py`.

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
  --embodiment-tag NEW_EMBODIMENT
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
