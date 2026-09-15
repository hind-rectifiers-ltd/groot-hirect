# Deployment Guide — ONNX / TensorRT + On-Robot Inference

End-to-end directions to **export ONNX**, optionally **build TensorRT engines**, and **run inference on the robot** for the 3-camera humanoid pipeline (`NEW_EMBODIMENT`).

Related docs:

- Data / train / PyTorch server flow: [`docs/Record_Flow_README.md`](Record_Flow_README.md)
- Upstream deployment reference (LIBERO / platforms): [`scripts/deployment/README.md`](../scripts/deployment/README.md)

---

## Paths at a glance

| Path | What you get | Use when |
|------|----------------|----------|
| **A. PyTorch ZMQ server** | Live policy on robot via `run_gr00t_server.py` + `policy_client_3cam.py` | Default closed-loop deploy (no ONNX required) |
| **B. ONNX only** | `.onnx` files under an output dir | You need portable graphs / TRT input |
| **C. ONNX → TensorRT** | GPU engines + faster E2E | Edge speedup after PyTorch path works |

Do **not** skip Path A validation. Fix finite actions (no `nan`) on PyTorch first, then export ONNX/TRT.

---

## 0) Prerequisites

From repo root, set the same env vars as training (adjust paths to your machine):

```bash
export REPO_ID=hirect_humanoid/pickplace_3cam
export LEROBOT_ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
export DS="${LEROBOT_ROOT}/${REPO_ID}"
export FT_OUT=./outputs/gr00t_custom_3cam
export CKPT="$(ls -d "${FT_OUT}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
echo "CKPT=${CKPT}"

export DEPLOY_OUT=./gr00t_trt_deployment_3cam
export EMBODIMENT=NEW_EMBODIMENT
export TASK='pick up the object and place it in the tray'   # must match meta/tasks.jsonl
```

Checklist before export/deploy:

- [ ] `"${CKPT}"` exists and contains `model.safetensors*` / `processor_config.json` / stats
- [ ] `"${DS}"` is the LeRobot v2.1 dataset used for fine-tune (needed to capture ONNX input shapes)
- [ ] `record/custom_3cam_config.py` was used at train time for `NEW_EMBODIMENT`
- [ ] GPU deps installed (`uv sync` on dGPU; Orin/Thor/Spark use platform install scripts — see `scripts/deployment/README.md`)

---

## 1) Path A — Deploy on robot (PyTorch, recommended first)

This is the closed-loop stack used with RobStride + 3 USB cameras.

### 1.1 Start policy server (GPU machine / Jetson)

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${CKPT}" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

Keep this process running. Confirm no CUDA errors on first client query.

> `--modality-config-path` on this server entrypoint applies to **ReplayPolicy** only. For a fine-tuned checkpoint, modality comes from the saved processor inside `"${CKPT}"`.

### 1.2 Read-only check (cameras + live qpos, **no motor writes**)

On the robot host (same LAN / localhost):

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "${TASK}" \
  --robot robstride \
  --use-usb-camera-ports \
  --camera-fps 30 \
  --image-height 640 \
  --image-width 640 \
  --dry-run-robstride \
  --control-mode chunk \
  --rate-hz 30 \
  --max-steps 20
```

Pass criteria:

- Log shows `RobStride READ-ONLY`
- `target[0]` is **finite** (not `+nan`)
- `|target[0] - qpos|` is small at home

If you see `policy action chunk contains non-finite values`, stop — fix checkpoint / stats / cameras / server before ONNX or live writes. See troubleshooting in `Record_Flow_README.md` §8 and the NaN notes below.

### 1.3 Live hardware

Only after targets are finite:

```bash
uv run python record/policy_client_3cam.py \
  --host localhost \
  --port 5555 \
  --task "${TASK}" \
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

Safer first pass: `--policy-ramp-max-speed 1.5 --action-smoothing-alpha 0.7 --max-target-step-arm 0.05`.

Match recording: `--rate-hz 30`, `--camera-fps 30`, 640×640, same `record/camera_ports.json`, exact task string.

---

## 2) Path B — Export ONNX files

ONNX export needs a **checkpoint** and a **LeRobot dataset** (shapes are captured from a real sample).

### 2.1 ONNX only (export step)

```bash
mkdir -p "${DEPLOY_OUT}/onnx"

uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${DEPLOY_OUT}/onnx" \
  --export-mode full_pipeline \
  --batch-size 1 \
  --precision bf16
```

**Export modes:**

| `--export-mode` | What is written | Typical platform |
|-----------------|-----------------|------------------|
| `full_pipeline` | ViT + LLM + VL self-attn + state/action encoders + DiT + action decoder | dGPU, Thor, Spark |
| `action_head` | 4 action-head components (backbone stays PyTorch) | Intermediate / debug |
| `dit_only` | DiT only | Jetson Orin (TRT 10.3 limitation) |

**Orin:** prefer `--export-mode dit_only` (or `action_head` for verify). Full backbone TRT is not supported on Orin TRT 10.3.

### 2.2 Where the ONNX files land

After a successful export, inspect:

```bash
ls -lh "${DEPLOY_OUT}/onnx"
cat "${DEPLOY_OUT}/onnx/export_metadata.json"
```

Typical `full_pipeline` artifacts (names may include precision tags such as `bf16`):

- `vit_*.onnx`
- `llm_*.onnx`
- `vl_self_attention.onnx`
- `state_encoder.onnx`
- `action_encoder.onnx`
- `dit_*.onnx`
- `action_decoder.onnx`
- `export_metadata.json`

These `.onnx` files are what you copy to another machine or feed into the TensorRT builder.

---

## 3) Path C — ONNX → TensorRT (recommended via unified pipeline)

`build_trt_pipeline.py` runs **export → build engines → verify → benchmark** in one command (or a subset of steps).

### 3.1 Full pipeline (dGPU / Thor / Spark)

```bash
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${DEPLOY_OUT}" \
  --export-mode full_pipeline \
  --batch-size 1 \
  --steps all
```

Outputs:

| Location | Contents |
|----------|----------|
| `${DEPLOY_OUT}/onnx/` | ONNX graphs |
| `${DEPLOY_OUT}/engines/` | TensorRT `.engine` files (GPU-architecture specific) |
| `${DEPLOY_OUT}/pipeline.log` | Verbose log |

### 3.2 Export + build only

```bash
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${DEPLOY_OUT}" \
  --export-mode full_pipeline \
  --steps export,build
```

### 3.3 Orin (DiT-only)

```bash
source .venv/bin/activate
source scripts/activate_orin.sh   # bare-metal Orin

uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${DEPLOY_OUT}" \
  --export-mode dit_only \
  --steps export,build,verify
```

Engines are **not portable** across GPU architectures (e.g. Orin ≠ dGPU). Rebuild on the target robot GPU.

---

## 4) Validate ONNX/TRT offline (before robot)

### 4.1 PyTorch open-loop (sanity)

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 16 \
  --steps 300
```

Predictions should be finite and track GT reasonably.

### 4.2 TensorRT open-loop

**Full pipeline (dGPU / Thor / Spark):**

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 1 2 \
  --inference-mode trt_full_pipeline \
  --trt-engine-path "${DEPLOY_OUT}/engines" \
  --action-horizon 16
```

**DiT-only (Orin):**

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path "${CKPT}" \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --inference-mode tensorrt \
  --trt-engine-path "${DEPLOY_OUT}/engines" \
  --action-horizon 16
```

### 4.3 Benchmark

```bash
uv run python scripts/deployment/benchmark_inference.py \
  --model-path "${CKPT}" \
  --trt-engine-path "${DEPLOY_OUT}/engines" \
  --trt-mode n17_full_pipeline
```

On Orin use DiT-only timing (see `scripts/deployment/README.md`).

---

## 5) On-robot inference after TRT build

Today’s **hardware client** (`policy_client_3cam.py`) talks to a **ZMQ PolicyServer**. The stock `run_gr00t_server.py` loads **PyTorch** `Gr00tPolicy` (it does not take `--trt-engine-path`).

Practical options:

1. **Keep Path A (PyTorch server)** for closed-loop robot control — simplest and already wired to cameras / RobStride.
2. **Use TRT offline** (`standalone_inference_script.py`) for speed/accuracy checks on the robot GPU.
3. **Custom integration**: load engines via `scripts/deployment/trt_model_forward.py` (`setup_tensorrt_engines`) into a `Gr00tPolicy`, then serve that policy with `PolicyServer` (advanced; not a one-liner in this repo).

Recommended robot day workflow:

```text
1. Preview cameras     → record/preview_three_cameras.py
2. Start PyTorch server → run_gr00t_server.py  (--model-path CKPT)
3. Dry-run client       → policy_client_3cam.py --dry-run-robstride
4. Live client          → policy_client_3cam.py --apply-actions
5. (Optional) TRT       → build on-device engines; validate with standalone_inference_script.py
```

Copy artifacts to the robot (example):

```bash
# From training / export machine
rsync -avP "${CKPT}/" robot:/path/to/ckpt/
rsync -avP "${DEPLOY_OUT}/onnx/" robot:/path/to/deploy/onnx/
# Prefer rebuilding engines ON the robot GPU rather than copying .engine files
```

---

## 6) Platform install (Jetson / Spark)

| Platform | Install | Activate each shell |
|----------|---------|---------------------|
| dGPU | `uv sync` | `source .venv/bin/activate` |
| Jetson Orin | `bash scripts/deployment/orin/install_deps.sh` | `source .venv/bin/activate && source scripts/activate_orin.sh` |
| Jetson Thor | `bash scripts/deployment/thor/install_deps.sh` | `source .venv/bin/activate && source scripts/activate_thor.sh` |
| DGX Spark | `bash scripts/deployment/spark/install_deps.sh` | `source .venv/bin/activate && source scripts/activate_spark.sh` |

Docker profiles: see `scripts/deployment/README.md` (`docker/build.sh --profile=orin|thor|spark`).

---

## 7) Quick command cheat sheet

```bash
# --- ONNX only ---
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path "${CKPT}" --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${DEPLOY_OUT}/onnx" \
  --export-mode full_pipeline

# --- ONNX + TRT engines ---
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path "${CKPT}" --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${DEPLOY_OUT}" \
  --export-mode full_pipeline --steps export,build

# --- Robot: server ---
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${CKPT}" --embodiment-tag NEW_EMBODIMENT \
  --device cuda --host 0.0.0.0 --port 5555

# --- Robot: read-only client ---
uv run python record/policy_client_3cam.py \
  --host localhost --port 5555 --task "${TASK}" \
  --robot robstride --use-usb-camera-ports --dry-run-robstride \
  --control-mode chunk --rate-hz 30 --max-steps 20

# --- Robot: live client ---
uv run python record/policy_client_3cam.py \
  --host localhost --port 5555 --task "${TASK}" \
  --robot robstride --use-usb-camera-ports --apply-actions \
  --control-mode chunk --rate-hz 30 \
  --policy-ramp-max-speed 2.5
```

---

## 8) Troubleshooting

### ONNX export fails

- Dataset path must be valid LeRobot v2.1 with at least one episode (`"${DS}"`).
- Pass `--embodiment-tag NEW_EMBODIMENT` if auto-detect finds multiple tags.
- GPU OOM: close other processes; try `--export-mode action_head` or `dit_only` first.
- See also `scripts/deployment/README.md` → Troubleshooting.

### TRT engine build fails

- Rebuild on the **same** GPU you will run on.
- Reduce `--workspace` (e.g. `4096`).
- Orin: LLM/backbone engine failure is expected — use `dit_only`.

### Policy returns all `nan` on robot

This is a **server/model/observation** issue, not ONNX itself:

1. Confirm PyTorch open-loop on `"${DS}"` is finite.
2. Confirm `stats.json` in ckpt/dataset has finite action stats.
3. Confirm cameras are not black (`preview_three_cameras.py`).
4. Use `--dry-run-robstride` until `target[0]` is finite; never `--apply-actions` while NaN.

### Cameras fall back to 15 FPS

Same as recording: 30 Hz control reuses the latest frame. Prefer USB 3 for true 30 FPS; not a blocker for ONNX export.

---

## 9) File map

| Script | Role |
|--------|------|
| `scripts/deployment/export_onnx_n1d7.py` | Export ONNX graphs |
| `scripts/deployment/build_trt_pipeline.py` | Export + TRT build + verify + benchmark |
| `scripts/deployment/build_tensorrt_engine.py` | Low-level engine builder (used by pipeline) |
| `scripts/deployment/standalone_inference_script.py` | Offline PyTorch / TRT trajectory eval |
| `scripts/deployment/benchmark_inference.py` | Latency breakdown |
| `gr00t/eval/run_gr00t_server.py` | Live ZMQ policy server (PyTorch) |
| `record/policy_client_3cam.py` | Robot client (cameras + RobStride) |
