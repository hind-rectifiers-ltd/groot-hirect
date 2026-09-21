# Recording + Fine-tune + Real-Hardware Validation (3-Cam, 16-DoF follower)

This folder contains the exact workflow used for a custom 3-camera humanoid setup:

- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`
- Leader: `12` Dynamixels (`5 arm + 1 gripper` per side)
- Follower / recorded joints: `16` DoF (`7 arm + 1 gripper` per side); missing leader wrist_roll/yaw map to follower `0`
- Raw capture format: `episode_XXXXXX.hdf5`
- Training format: GR00T-compatible LeRobot v2.1
- Embodiment tag: `NEW_EMBODIMENT` (registered by `record/custom_3cam_config.py`)

Use this as an actionable runbook from data collection to real robot rollout.

**Multiple skills (task 2+):** see [`docs/Multi_record_flow_guide.md`](Multi_record_flow_guide.md) for which env vars change per task, per-folder recording, staging, and one shared fine-tune.

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

Jetson (T7 mounted under `/media/jetson`):

```bash
export SSD=/media/jetson/T7/pick_place_v1
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

Verified mapping for this Jetson rig (from `--preview-all-cameras`):


| Role        | HDF5 / LeRobot key | USB port (`id_path_tag`)           | Typical node |
| ----------- | ------------------ | ---------------------------------- | ------------ |
| Head        | `cam_head`         | `platform-3610000_usb-usb-0_4_3`   | `/dev/video8` |
| Left wrist  | `cam_left_wrist`   | `platform-3610000_usb-usb-0_4_1`   | `/dev/video0` |
| Right wrist | `cam_right_wrist`  | `platform-3610000_usb-usb-0_4_2`   | `/dev/video4` |


The JSON on disk:

```json
{
  "cam_head": {"id_path_tag": "platform-3610000_usb-usb-0_4_3"},
  "cam_left_wrist": {"id_path_tag": "platform-3610000_usb-usb-0_4_1"},
  "cam_right_wrist": {"id_path_tag": "platform-3610000_usb-usb-0_4_2"}
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

> **Black panel with 3 cams open, but OK one-at-a-time:** All three cams on this Jetson currently share one **USB 2.0** hub (`lsusb -t` → Bus 01 @ 480M). Record_Flow targets **`--camera-fps 30`** (same clock as `--teleop-rate 30` / `--dt 0.0333333`). If the hub cannot sustain 30 FPS, `USBCameraRig` **auto-falls back to 15 FPS** and prints a warning. For true **30 FPS** on Jetson: plug at least one camera into a **USB 3** port (Bus 02 @ 5000M–10000M), re-run `--preview-all-cameras`, update `camera_ports.json`, then verify with `preview_three_cameras.py` (log should say `fps=30` with no fallback warning).

---

## 1) Record episodes (v2 flow input)

### Timing (important for GR00T)

Keep **motors, camera sampling, conversion fps, and deployment rate** on the same clock (**30 Hz / 30 fps** for this setup):


| Stage                                          | Flag                    | Value                            |
| ---------------------------------------------- | ----------------------- | -------------------------------- |
| Follower teleop                                | `--teleop-rate`         | `30`                             |
| Record loop (images + `qpos`/`action` logging) | `--dt`                  | `0.0333333` (~30 Hz)             |
| UVC camera capture                             | `--camera-fps`          | `30` (auto-falls back to 15 on saturated USB 2.0 hub) |
| Qpos filtering at 30 Hz                        | `--qpos-median-samples` | `2` (auto when teleop-rate ≥ 20) |
| LeRobot convert                                | `--fps`                 | `30`                             |
| Policy client                                  | `--rate-hz`             | `30`                             |


USB cameras should stream at **`--camera-fps 30`** so each `--dt` tick gets a fresh frame. Do **not** mix rates (e.g. `--dt 0.1` with `--teleop-rate 30`) — that misaligns vision and joints in training data. On this Jetson, all three cams on one USB 2.0 hub cannot sustain 30 FPS; move a cam to USB 3 (see [Camera setup](#camera-setup-usb-ports-not-devvideon)) before production recording.

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
  --camera-fps 30 \
  --qpos-median-samples 2 \
  --use-usb-camera-ports \
  --task "pick up the object and place it in the tray"
```

Notes:

- `--use-usb-camera-ports` (default on Linux) loads `record/camera_ports.json` — same mapping used at inference.
- `--teleop-rate 30`, `--dt 0.0333333`, `--camera-fps 30`, and convert `--fps 30` must all match. If the log shows a USB 2.0 fallback to 15 FPS, move a camera to USB 3 before production demos.
- At ≥ 20 Hz the recorder auto-uses **2 back-to-back median reads** (no gap), zero-dropout filter only, parallel CAN reads, and encoder `feedback` on commands. Safety mode defaults to **clamp** (use `--safety-abort` to disconnect on breach).
- After countdown, session is **idle** with teleop live: press `r` to start the first episode.
- `Esc` / `s` ends current episode and saves in the background; press `r` to start the next episode immediately (even while saving).
- `d` discards the current episode (no HDF5 written); next `r` reuses the same `episode_XXXXXX.hdf5` index.
- If you stop an episode while the previous save is still running, the recorder waits briefly before writing the new file (max 1 save at a time).
- `Ctrl+C` ends current episode (saves if any frames) and quits the session.
- RobStride/teleop stays connected across episodes — no need to restart the script between takes.
- Re-running the same command still auto-increments `episode_XXXXXX.hdf5` from existing files.
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
# Half speed: --speed 0.5  |  2x: --speed 2
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

Viewer keys: Space pause, `n`/`p` episode, `,`/`.` skip ±5 frames, `r` restart, `q` quit. `--speed` multiplies playback rate (default `1.0`).

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
  --state-dim 16 \
  --action-dim 16
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

**Cameras + live arm reads, no motor writes** (preferred NaN / policy check):

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "pick up the object and place it in the tray" \
  --robot robstride \
  --use-usb-camera-ports \
  --camera-fps 30 \
  --image-height 640 \
  --image-width 640 \
  --dry-run-robstride \
  --control-mode chunk \
  --rate-hz 30 \
  --max-steps 10
```

(Same mode if you omit `--apply-actions`: CAN opens read-only, encoders + cameras feed the policy, nothing is commanded.)

Look at `target[0]`: must be finite numbers near home (~0), **not** `+nan`. Cameras may fall back to 15 FPS on USB 2; the 30 Hz loop reuses the latest frame (same as recording). Fix USB 3 when you can.

**Cameras only** (no CAN at all): `--robot usb_cam` instead of `--robot robstride --dry-run-robstride`.

**Live hardware** (only after targets are finite):

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "pick up the object and place it in the tray" \
  --robot robstride \
  --use-usb-camera-ports \
  --camera-fps 30 \
  --image-height 640 \
  --image-width 640 \
  --apply-actions \
  --control-mode chunk \
  --rate-hz 30 \
  --policy-ramp-max-speed 2.5 \
  --action-smoothing-alpha 0.5 \
  --gripper-smoothing-alpha 0.7 \
  --chunk-blend-steps 12
```

`--use-usb-camera-ports` is on by default on Linux; it reads `record/camera_ports.json` so inference sees the same head / left / right views as training.

**Must match recording/training:** `--rate-hz 30`, `--camera-fps 30`, 640×640 images, same camera ports, and the **exact** `--task` string from `meta/tasks.jsonl`.

Safer first live pass (same rate match, slower arms):

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "pick up the object and place it in the tray" \
  --robot robstride \
  --use-usb-camera-ports \
  --camera-fps 30 \
  --apply-actions \
  --control-mode chunk \
  --rate-hz 30 \
  --policy-ramp-max-speed 1.5 \
  --action-smoothing-alpha 0.7 \
  --chunk-blend-steps 8 \
  --max-target-step-arm 0.05
```

Useful flags:

- `--control-mode chunk` (default): run the 16-step horizon before re-inferring.
- `--rate-hz 30`: match record (`--teleop-rate 30`, `--dt 0.0333333`) and convert (`--fps 30`).
- `--policy-ramp-max-speed`: MIT slew cap (try 1.5–2.5 for smoother arms).
- `--chunk-blend-steps 12`: soften replan boundaries.
- `--preview-cameras` on `record_episodes_3cam.py` or `preview_three_cameras.py` if views look swapped mid-session.
- `--no-use-usb-camera-ports --video-cam-head …` only for legacy numeric overrides.
- Before `--apply-actions`, run once with `--dry-run-robstride` and check logs: `qpos` vs `target[0]` should be close at home.

Real-robot validation checklist:

- `preview_three_cameras.py` labels match physical cameras.
- Task string matches `meta/tasks.jsonl` / training text exactly.
- Follower at home pose (~0 rad); `qpos` stable in logs (no periodic garbage reads).
- At step 0, `|target[0] - qpos|` should be small; large gaps mean policy/data/camera mismatch, not rate settings.
- Start with low `--policy-ramp-max-speed` and clear workspace before full task rollout.

---

## 9) Optional edge deployment path (ONNX/TensorRT)

Full directions (export ONNX → TensorRT → on-robot inference for `NEW_EMBODIMENT`): see **[`docs/DEPLOYMENT.md`](DEPLOYMENT.md)**.

Quick pointers:

```bash
# ONNX only
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ./gr00t_trt_deployment_3cam/onnx \
  --export-mode full_pipeline

# ONNX + TensorRT engines (unified pipeline)
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ./gr00t_trt_deployment_3cam \
  --export-mode full_pipeline

# Benchmark
uv run python scripts/deployment/benchmark_inference.py \
  --model-path "${CKPT}" \
  --trt-engine-path ./gr00t_trt_deployment_3cam/engines \
  --trt-mode n17_full_pipeline
```

On Jetson Orin use `--export-mode dit_only`. Closed-loop robot control still uses `run_gr00t_server.py` + `policy_client_3cam.py` (Path A in `DEPLOYMENT.md`).

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
9. Optional edge deploy (ONNX/TRT): follow [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).

