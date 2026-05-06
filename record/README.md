# Recording + Fine-tune + Real-Hardware Validation (3-Cam, 12-DoF)

This folder contains the exact workflow used for a custom 3-camera humanoid setup:

- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`
- Joint layout: `12` DoF (`left_arm[5] + left_gripper[1] + right_arm[5] + right_gripper[1]`)
- Raw capture format: `episode_XXXXXX.hdf5`
- Training format: GR00T-compatible LeRobot v2.1
- Embodiment tag: `NEW_EMBODIMENT` (registered by `record/custom_3cam_config.py`)

Use this as an actionable runbook from data collection to real robot rollout.

## Tools in this folder

- `record/record_episodes_3cam.py`: records raw HDF5 episodes.
- `record/visualize_recorded_episodes.py`: validates multi-camera episodes and joint traces.
- `record/convert_3cam_to_groot_lerobot.py`: converts raw episodes to LeRobot (v3 layout first).
- `record/custom_3cam_config.py`: registers modality config for `NEW_EMBODIMENT`.
- `record/policy_client_3cam.py`: client for GR00T server (USB cam / demo / RobStride backends).

---

## 0) One-time setup and naming

Use a consistent task sentence end-to-end (recording, training, inference).  
For this project, the task text used in demos is:

`pick up the object and place it in the tray`

Set convenient variables from repo root:

```bash
export RAW_DIR=./record/pickplace
export REPO_ID=hirect_humanoid/pickplace_3cam
export LEROBOT_ROOT=/home/yash/.cache/huggingface/lerobot
export DS=${LEROBOT_ROOT}/${REPO_ID}
export FT_OUT=./outputs/gr00t_custom_3cam
```

---

## 1) Record episodes (v2 flow input)

Check camera IDs first:

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir "${RAW_DIR}" \
  --test-cameras \
  --test-cameras-max 16
```

Record with direct teleop + RobStride follower + 3 cameras:

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir "${RAW_DIR}" \
  --robot direct_teleop \
  --leader-port /dev/ttyACM0 \
  --leader-baud 57600 \
  --teleop-rate 10 \
  --video-cam-head 12 \
  --video-cam-left-wrist 2 \
  --video-cam-right-wrist 1 \
  --task "pick up the object and place it in the tray"
```

Notes:

- Recording starts after the countdown and `Start teleoperating now.` message.
- `Ctrl+C` ends current episode and saves what is captured.
- Re-running the same command auto-increments `episode_XXXXXX.hdf5`.

---

## 2) Visualize recorded episodes before conversion

Inspect one session:

```bash
uv run python record/visualize_recorded_episodes.py --data-dir "${RAW_DIR}"
```

Inspect recursively under `record/`:

```bash
uv run python record/visualize_recorded_episodes.py --data-dir ./record --recursive
```

Export one preview clip:

```bash
uv run python record/visualize_recorded_episodes.py \
  --data-dir "${RAW_DIR}" \
  --episode 0 \
  --save-video /tmp/preview_pickplace.mp4
```

Checklist before conversion:

- Camera streams are synchronized and not swapped.
- Task text overlay is correct.
- `qpos` and `action` traces look smooth and physically plausible.

---

## 3) Convert to LeRobot and then to GR00T-compatible v2.1

### 3.1 Raw HDF5 -> LeRobot (current converter writes v3 layout first)

```bash
uv run python record/convert_3cam_to_groot_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id "${REPO_ID}" \
  --fps 30 \
  --state-dim 12 \
  --action-dim 12
```

### 3.2 LeRobot v3 -> v2.1 (required for GR00T loader)

```bash
uv run python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id "${REPO_ID}" \
  --root "${LEROBOT_ROOT}"
```

Restore modality metadata from backup (`*_v3.0`) because v3->v2 conversion does not carry it:

```bash
cp "${DS}_v3.0/meta/modality.json" "${DS}/meta/modality.json"
```

Optional (if present):

```bash
test -f "${DS}_v3.0/meta/relative_stats.json" && cp "${DS}_v3.0/meta/relative_stats.json" "${DS}/meta/"
```

---

## 4) Generate dataset statistics (required)

`NEW_EMBODIMENT` is custom and must be registered by loading `record/custom_3cam_config.py`.

```bash
uv run python gr00t/data/stats.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py
```

Expected files:

- `${DS}/meta/stats.json`
- `${DS}/meta/relative_stats.json` (may be mostly empty with absolute-action config)

Rerun this whenever modality/action definitions change.

---

## 5) Fine-tune GR00T N1.7

Single-GPU command used in this setup:

```bash
export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "${FT_OUT}" \
  --max-steps 10000 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --global-batch-size 32 \
  --dataloader-num-workers 4
```

Checkpoint example:

- `./outputs/gr00t_custom_3cam/checkpoint-10000`

Use `./outputs` (inside repo) instead of `/tmp` so checkpoints are persistent.

---

## 6) Open-loop sanity evaluation (recommended)

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path ./outputs/gr00t_custom_3cam/checkpoint-10000 \
  --traj-ids 0 1 2 \
  --action-horizon 16 \
  --steps 300
```

If predictions diverge strongly from GT, improve data quality/coverage before hardware rollout.

---

## 7) Run inference server with fine-tuned checkpoint

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path ./outputs/gr00t_custom_3cam/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

---

## 8) Validate on real hardware (`policy_client_3cam.py`)

Run RobStride client against server:

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "pick up the object and place it in the tray" \
  --robot robstride \
  --video-cam-head 12 \
  --video-cam-left-wrist 2 \
  --video-cam-right-wrist 1 \
  --apply-actions
```

Useful tuning flags:

- `--policy-ramp-max-speed 1.5` to make initial motion gentler.
- `--ramp-from-feedback` only if needed (can introduce oscillation in some setups).
- `--negate-a12-indices 1` to test a sign flip for the second left-arm joint (motor 3) if motion direction is inverted.

Real-robot validation checklist:

- Start with small-speed ramp and clear workspace.
- Confirm `qpos` is stable (no frequent all-zero reads).
- Verify first motion direction and joint limits before full task rollout.
- Keep task text exactly aligned with training text.

---

## 9) Optional edge deployment path (ONNX/TensorRT)

Export ONNX:

```bash
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path ./outputs/gr00t_custom_3cam/checkpoint-10000
```

Build TensorRT pipeline:

```bash
uv run python scripts/deployment/build_trt_pipeline.py
```

Benchmark:

```bash
uv run python scripts/deployment/benchmark_inference.py
```

---

## End-to-end quick checklist

1. Record demos with correct camera IDs and consistent task text.
2. Visualize raw episodes and remove bad captures.
3. Convert raw -> LeRobot -> v2.1 and restore `meta/modality.json`.
4. Generate stats with `record/custom_3cam_config.py`.
5. Fine-tune to `./outputs/gr00t_custom_3cam`.
6. Run open-loop eval.
7. Start GR00T server from `./outputs/.../checkpoint-10000`.
8. Validate closed-loop on robot with `record/policy_client_3cam.py`.

