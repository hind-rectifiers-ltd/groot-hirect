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
- `record/usb_cameras.py` + `record/camera_ports.json`: stable USB port → camera role mapping.
- `record/preview_three_cameras.py`: live labeled preview to confirm HEAD / wrist mapping.

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
export CKPT="${FT_OUT}/checkpoint-12000"
# Tip: auto-pick the latest checkpoint after training:
#   export CKPT="$(ls -d "${FT_OUT}"/checkpoint-* | sort -V | tail -n1)"
```

If internal disk is tight, keep raw episodes, LeRobot output, HF cache, and checkpoints on an external drive (adjust `SSD` to your mount point):

```bash
export SSD=/media/yash/T7/pick_place_v1
export RAW_DIR="${SSD}/pick_place"
export REPO_ID=hirect_humanoid/pickplace_3cam
export HF_LEROBOT_HOME="${SSD}/lerobot"
export LEROBOT_ROOT="${HF_LEROBOT_HOME}"
export DS="${LEROBOT_ROOT}/${REPO_ID}"
export FT_OUT="${SSD}/outputs/gr00t_custom_3cam"
export HF_HOME="${SSD}/huggingface"
export CKPT="${FT_OUT}/checkpoint-10000"
# Tip: auto-pick the latest checkpoint after training:
#   export CKPT="$(ls -d "${FT_OUT}"/checkpoint-* | sort -V | tail -n1)"
```

`HF_LEROBOT_HOME` is what the LeRobot converter uses; `HF_HOME` routes large Hugging Face / Transformers downloads (e.g. base model) to the SSD. `FT_OUT` is the training output directory and `CKPT` points to one specific checkpoint inside it — use `"${CKPT}"` everywhere the README needs a checkpoint path. Update the checkpoint number whenever you train further (or use the `ls`-based tip above).

### Camera setup (USB ports, not `/dev/videoN`)

On Linux, **do not rely on numeric camera indices** (`/dev/video0`, `/dev/video8`, …) — they change across reboots. Instead, cameras are pinned by **physical USB port** in `record/camera_ports.json` and resolved automatically when `--use-usb-camera-ports` is set (default on Linux for record + policy client).

Verified mapping for this rig:


| Role        | HDF5 / LeRobot key | USB port (`id_path_tag`)   | Typical node  |
| ----------- | ------------------ | -------------------------- | ------------- |
| Head        | `cam_head`         | `pci-0000_00_14_0-usb-0_8` | `/dev/video8` |
| Left wrist  | `cam_left_wrist`   | `pci-0000_00_14_0-usb-0_9` | `/dev/video6` |
| Right wrist | `cam_right_wrist`  | `pci-0000_00_14_0-usb-0_3` | `/dev/video0` |


The JSON on disk:

```json
{
  "cam_head": {"id_path_tag": "pci-0000_00_14_0-usb-0_8"},
  "cam_left_wrist": {"id_path_tag": "pci-0000_00_14_0-usb-0_9"},
  "cam_right_wrist": {"id_path_tag": "pci-0000_00_14_0-usb-0_3"}
}
```

**Before every record or inference session** (or after moving USB cables), confirm labels with a live preview:

```bash
uv run python record/preview_three_cameras.py
```

Wave each physical camera; the **HEAD**, **LEFT WRIST**, and **RIGHT WRIST** banners must match the view. Press **Esc** to quit.

If a cable moved or you are setting up a new machine:

```bash
# Discover instance= all cameras at a time
uv run python record/preview_three_cameras.py --preview-cameras
# Discover instance= ids one camera at a time (Space = next, Esc = quit)
uv run python record/preview_three_cameras.py --preview-all-cameras

# Text-only listing
uv run python record/record_episodes_3cam.py --list-cameras-working
```

Edit `record/camera_ports.json`, then run `preview_three_cameras.py` again until correct.

Override a single role without editing JSON: `--video-cam-head /dev/video8`. Disable USB pinning: `--no-use-usb-camera-ports` plus explicit `--video-cam-*` indices.

---

## 1) Record episodes (v2 flow input)

### Timing (important for GR00T)

Keep **motors, camera sampling, conversion fps, and deployment rate** on the same clock (**30 Hz / 30 fps** for this setup):


| Stage                                          | Flag                    | Value                            |
| ---------------------------------------------- | ----------------------- | -------------------------------- |
| Follower teleop                                | `--teleop-rate`         | `30`                             |
| Record loop (images + `qpos`/`action` logging) | `--dt`                  | `0.0333333` (~30 Hz)             |
| Qpos filtering at 30 Hz                        | `--qpos-median-samples` | `2` (auto when teleop-rate ≥ 20) |
| LeRobot convert                                | `--fps`                 | `30`                             |
| Policy client                                  | `--rate-hz`             | `30`                             |


USB cameras may run faster internally; the recorder grabs **one frame per `--dt` tick**. Do **not** mix rates (e.g. `--dt 0.1` with `--teleop-rate 30`) — that misaligns vision and joints in training data.

Validate CAN on the robot before your first 30 Hz session (arm still, both buses up):

```bash
uv run python scripts/benchmark_follower_hz.py \
  --rates 30 --mode record --median-samples 2 --duration-s 20
```

Expect **PASS** (~22 ms/tick). See [Validate CAN before recording](#validate-can-before-recording) for the full sweep.

Confirm cameras first (`preview_three_cameras.py` — see [Camera setup](#camera-setup-usb-ports-not-devvideon)).

Record with direct teleop + RobStride follower + 3 cameras (**30 Hz / 30 fps**):

```bash
uv run python record/record_episodes_3cam.py \
  --output-dir "${RAW_DIR}" \
  --robot direct_teleop \
  --leader-port /dev/ttyACM0 \
  --leader-baud 57600 \
  --teleop-rate 30 \
  --dt 0.0333333 \
  --qpos-median-samples 2 \
  --use-usb-camera-ports \
  --task "pick up the object and place it in the tray"
```

Notes:

- `--use-usb-camera-ports` (default on Linux) loads `record/camera_ports.json` — same mapping used at inference.
- `--teleop-rate 30`, `--dt 0.0333333`, and convert `--fps 30` must all match.
- At ≥ 20 Hz the recorder auto-uses **2 back-to-back median reads** (no gap), zero-dropout filter only, parallel CAN reads, and `feedback12` on commands.
- Recording starts after the countdown and `Start teleoperating now.` message.
- `Ctrl+C` ends current episode and saves what is captured.
- Re-running the same command auto-increments `episode_XXXXXX.hdf5`.
- After a test episode, confirm HDF5 `fps` ≈ 30 (cameras must keep up with `--dt`).
- HDF5 stores `cam_head` / `cam_left_wrist` / `cam_right_wrist` by role name (not `/dev/videoN`), so conversion and training are unaffected by node renumbering as long as recording used the correct port mapping.

### Validate CAN before recording

Optional full sweep (read → teleop → record):

```bash
uv run python scripts/benchmark_follower_hz.py --rates 10 20 30 --mode read --duration-s 20
uv run python scripts/benchmark_follower_hz.py --rates 10 20 30 --mode teleop --duration-s 20
uv run python scripts/benchmark_follower_hz.py --rates 10 20 30 --mode record --median-samples 1 --duration-s 15
```

Each trial prints **PASS/FAIL** (actual Hz ≥ 95% of target, ≤5% missed deadlines, ≤1% read failures, no zero dropouts).

Do **not** use `--qpos-median-samples 1` or `3` at 30 Hz (1 = no spike rejection; 3 = too slow). After each episode, check for `qpos sanitizer holds` — **re-record if `zero_dropouts > 0`**.

**Lower rate fallback:** for debugging only, use `--teleop-rate 10 --dt 0.1`, convert `--fps 10`, deploy `--rate-hz 10`.

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

- Camera preview was verified (`preview_three_cameras.py`) before recording — head / wrists not swapped in HDF5.
- HDF5 `fps` attribute is ~30 (from `--dt 0.0333333`).
- Task text overlay is correct.
- `qpos` and `action` traces look smooth and physically plausible (no repeated joint rows between image changes).

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

`--fps` must match the record loop rate (`1 / --dt`, i.e. `30` when `--dt 0.0333333`).

Conversion reads HDF5 image keys (`cam_head`, `cam_left_wrist`, `cam_right_wrist`) — no camera device flags needed here. If wrist cameras look swapped in `visualize_recorded_episodes.py`, fix `camera_ports.json`, re-record; do not patch labels in the converter.

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

Training uses the LeRobot dataset on disk (video + state + action). Camera USB mapping is **not** involved at train time — only data recorded with the correct `camera_ports.json` mapping matters.

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
  --global-batch-size 4 \
  --dataloader-num-workers 0 \
  --gradient-checkpointing
```

If the host runs out of RAM during training, resume from the same `--output-dir` with a smaller batch and no dataloader workers:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --num-gpus 1 \
  --output-dir "${FT_OUT}" \
  --max-steps 10000 \
  --save-steps 2000 \
  --global-batch-size 2 \
  --dataloader-num-workers 0 \
  --gradient-checkpointing \
  --shard-size 512
```

Checkpoint example:

- `"${FT_OUT}/checkpoint-12000"` (i.e. `"${CKPT}"` from the export block)

Use `./outputs` (inside repo) or the SSD path instead of `/tmp` so checkpoints are persistent.

---

## 6) Open-loop sanity evaluation (recommended)

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "${CKPT}" \
  --traj-ids 0 1 2 \
  --action-horizon 16 \
  --steps 300
```

If predictions diverge strongly from GT, improve data quality/coverage before hardware rollout.

---

## 7) Run inference server with fine-tuned checkpoint

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${CKPT}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

Keep this running; start the policy client in [section 8](#8-validate-on-real-hardware-policy_client_3campy).

---

## 8) Validate on real hardware (`policy_client_3cam.py`)

Use the **same** `record/camera_ports.json` as recording. Preview before the first inference run:

```bash
uv run python record/preview_three_cameras.py
```

With the server from [section 7](#7-run-inference-server-with-fine-tuned-checkpoint) already running:

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "pick up the object and place it in the tray" \
  --robot robstride \
  --use-usb-camera-ports \
  --apply-actions \
  --control-mode chunk \
  --rate-hz 30 \
  --policy-ramp-max-speed 2.5 \
  --action-smoothing-alpha 0.5 \
  --chunk-blend-steps 12
```

`--use-usb-camera-ports` is on by default on Linux; it reads `record/camera_ports.json` so inference sees the same head / left / right views as training.

Useful flags:

- `--control-mode chunk` (default): run the 16-step horizon before re-inferring.
- `--rate-hz 30`: match record (`--teleop-rate 30`, `--dt 0.0333333`) and convert (`--fps 30`).
- `--policy-ramp-max-speed`: MIT slew cap (try 2.0–2.5 for smoother arms).
- `--chunk-blend-steps 12`: soften replan boundaries.
- `--preview-cameras` on `record_episodes_3cam.py` or `preview_three_cameras.py` if views look swapped mid-session.
- `--no-use-usb-camera-ports --video-cam-head …` only for legacy numeric overrides.

Real-robot validation checklist:

- `preview_three_cameras.py` labels match physical cameras.
- Task string matches `meta/tasks.jsonl` / training text exactly.
- Follower at home pose (~0 rad); `qpos` stable in logs (no periodic garbage reads).
- Start with low `--policy-ramp-max-speed` and clear workspace before full task rollout.

---

## 9) Optional edge deployment path (ONNX/TensorRT)

Export ONNX:

```bash
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path "${CKPT}"
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

1. Verify cameras: `uv run python record/preview_three_cameras.py` (HEAD / LEFT WRIST / RIGHT WRIST correct via `record/camera_ports.json`).
2. Record demos at **30 Hz / 30 fps** (`--teleop-rate 30`, `--dt 0.0333333`, `--use-usb-camera-ports`, consistent task text).
3. Visualize raw episodes (`visualize_recorded_episodes.py`); remove bad captures.
4. Convert raw → LeRobot → v2.1 with `--fps 30`; restore `meta/modality.json`.
5. Generate stats with `record/custom_3cam_config.py`.
6. Fine-tune to `"${FT_OUT}"`; pick a checkpoint in `"${CKPT}"`.
7. Open-loop eval: `open_loop_eval.py` with `--model-path "${CKPT}"`.
8. Preview cameras again, start server (`run_gr00t_server.py`), then closed-loop `policy_client_3cam.py` with `--use-usb-camera-ports` and `--rate-hz 30`.

