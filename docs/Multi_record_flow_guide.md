# Multi-Record Flow Guide

Ordered runbook for **collecting several skills in separate folders**, then **converting and fine-tuning them together** on the same embodiment.

Concepts / open-loop scoring: [`MULTI_TASK_PIPELINE.md`](MULTI_TASK_PIPELINE.md)  
Skill names & episode targets: [`task_dictionary.md`](task_dictionary.md)  
Single-task command details (cameras, 30 Hz, safety): [`Record_Flow_README.md`](Record_Flow_README.md)

---

## What changes vs what stays fixed

| Variable / flag | Change per skill? | Example |
|---|---|---|
| `SKILL_ID` | **Yes** | `pick_place_tray` → `pick_place_bowl` |
| `TASK` / `--task` | **Yes** (exact sentence) | `…place it in the tray` → `…place it in the bowl` |
| `RAW_DIR` | **Yes** (own folder) | `…/raw/pick_place_tray` → `…/raw/pick_place_bowl` |
| Scene / objects | **Yes** | tray vs bowl setup |
| `REPO_ID` / `DS` | **Once** for the mixed train set | `hirect_humanoid/multitask_3cam` |
| `FT_OUT` / `CKPT` | **New** multi-task train dir | do not overwrite single-task ckpts |
| Cameras (`camera_ports.json`) | **No** | same ports |
| `--teleop-rate` / `--dt` / `--camera-fps` | **No** | keep **30** |
| `--embodiment-tag` / modality config | **No** | `NEW_EMBODIMENT` + `record/custom_3cam_config.py` |
| Leader port / baud | **No** | same hardware |

**Rule:** record text ≡ `tasks.jsonl` text ≡ inference `--task`. Never paraphrase casually.

---

## 0) One-time multi-task env (SSD layout)

From repo root (adjust `SSD` if needed):

```bash
export SSD=/media/yash/T7/pick_place_v1
export HF_HOME="${SSD}/huggingface"
export HF_LEROBOT_HOME="${SSD}/lerobot"
export LEROBOT_ROOT="${HF_LEROBOT_HOME}"

# Mixed LeRobot dataset + multi-task fine-tune output (shared across skills)
export REPO_ID=hirect_humanoid/multitask_3cam
export DS="${LEROBOT_ROOT}/${REPO_ID}"
export FT_OUT="${SSD}/outputs/gr00t_multitask_3cam"
export CKPT="$(ls -d "${FT_OUT}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"

mkdir -p "${SSD}/raw" "${HF_LEROBOT_HOME}" "${FT_OUT}"
```

Per-skill raw roots:

```text
${SSD}/raw/
  pick_place_tray/     # task 1 (you may already have this as pick_place/)
  pick_place_bowl/     # task 2
  …                    # more skill IDs from task_dictionary.md
```

If task-1 data still lives at `"${SSD}/pick_place"`, either leave it and point `RAW_DIR` there for that skill, or move/symlink into `"${SSD}/raw/pick_place_tray"`.

---

## 1) Freeze the dictionary for this round

Example: task 1 already done; now add **task 2**.

| Order | Skill ID | Canonical `--task` | Target episodes |
|---|---|---|---|
| 1 | `pick_place_tray` | `pick up the object and place it in the tray` | ~80 |
| 2 | `pick_place_bowl` | `pick up the object and place it in the bowl` | ~70 |

Keep counts within ~1.5× of each other so one skill does not dominate training.

---

## 2) Record skill N (repeat for each new task)

### 2.1 Switch variables for this skill

**Task 2 example:**

```bash
export SKILL_ID=pick_place_bowl
export TASK="pick up the object and place it in the bowl"
export RAW_DIR="${SSD}/raw/${SKILL_ID}"
mkdir -p "${RAW_DIR}"
```

**Next skill:** only change `SKILL_ID` + `TASK` (+ scene). Do **not** change cameras, rate, or embodiment.

### 2.2 Cameras (same as single-task)

```bash
uv run python record/preview_three_cameras.py
```

### 2.3 Record

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
  --task "${TASK}"
```

Keys: `r` start episode, `s`/`Esc` save, `d` discard, `Ctrl+C` quit.  
One episode = one full successful skill. Drop failures.

### 2.4 Visualize a few episodes

```bash
uv run python record/visualize_recorded_episodes.py \
  --data-dir "${RAW_DIR}"
```

Confirm task overlay matches `"${TASK}"`, cameras correct, joints smooth.

### 2.5 Then move to the next skill

```bash
# e.g. task 3 from task_dictionary.md
export SKILL_ID=place_on_marker
export TASK="pick up the object and place it on the marker"
export RAW_DIR="${SSD}/raw/${SKILL_ID}"
mkdir -p "${RAW_DIR}"
# re-run 2.2 → 2.4
```

---

## 3) Stage all skills into one convert folder

`convert_3cam_to_groot_lerobot.py` takes **one** `--raw-dir` and does not append to an existing LeRobot repo. Stage every skill’s HDF5s (unique names) into one directory, then convert once.

```bash
export STAGE_DIR="${SSD}/raw/_staging_multitask"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

i=0
for skill in pick_place_tray pick_place_bowl; do
  # adjust path if task-1 still lives elsewhere, e.g. "${SSD}/pick_place"
  src="${SSD}/raw/${skill}"
  if [[ ! -d "${src}" ]]; then
    echo "WARN: missing ${src}"; continue
  fi
  for f in "${src}"/episode_*.hdf5; do
    [[ -e "$f" ]] || continue
    ln -sf "$(realpath "$f")" \
      "${STAGE_DIR}/episode_$(printf '%06d' "${i}").hdf5"
    i=$((i + 1))
  done
done

echo "Staged ${i} episodes into ${STAGE_DIR}"
ls "${STAGE_DIR}" | head
export RAW_DIR="${STAGE_DIR}"
```

Task text comes from each HDF5 `task` attribute (set at record time). Staging must **not** overwrite that.

Add more skill folder names to the `for skill in …` list as you grow.

---

## 4) Convert → LeRobot v2.1 (once for the mix)

```bash
export REPO_ID=hirect_humanoid/multitask_3cam
export DS="${LEROBOT_ROOT}/${REPO_ID}"

# 4.1 HDF5 → LeRobot v3 (uses HF_LEROBOT_HOME)
HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
  scripts/lerobot_conversion/.venv/bin/python record/convert_3cam_to_groot_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id "${REPO_ID}" \
  --fps 30 \
  --state-dim 16 \
  --action-dim 16 \
  --overwrite
```

Use the conversion venv for `lerobot` (do not `uv add lerobot` into the main gr00t env). If that venv is missing: `GIT_LFS_SKIP_SMUDGE=1 uv sync --project scripts/lerobot_conversion`.

```bash
# 4.2 v3 → v2.1
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id "${REPO_ID}" \
  --root "${LEROBOT_ROOT}"

cp "${DS}_v3.0/meta/modality.json" "${DS}/meta/modality.json"
test -f "${DS}_v3.0/meta/relative_stats.json" && cp "${DS}_v3.0/meta/relative_stats.json" "${DS}/meta/"
```

### Verify language labels

```bash
# All skills must appear here with exact strings
cat "${DS}/meta/tasks.jsonl"
wc -l "${DS}/meta/episodes.jsonl" 2>/dev/null || ls "${DS}/meta/"
```

---

## 5) Dataset statistics

```bash
uv run python gr00t/data/stats.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py
```

---

## 6) Fine-tune **one** multi-task checkpoint

Train on the **mixed** `"${DS}"` (all skills). Use a **new** `"${FT_OUT}"` so you keep the old single-task checkpoints.

```bash
export FT_OUT="${SSD}/outputs/gr00t_multitask_3cam"
mkdir -p "${FT_OUT}"

export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "${FT_OUT}" \
  --max-steps 12000 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --global-batch-size 4 \
  --dataloader-num-workers 0 \
  --gradient-checkpointing
```

```bash
export CKPT="$(ls -d "${FT_OUT}"/checkpoint-* | sort -V | tail -n1)"
echo "CKPT=${CKPT}"
# Prefer an earlier ckpt if a later one has NaN weights (see open-loop / weight scan).
```

`launch_finetune.py` currently takes a **single** `--dataset-path`. That is why this guide **stages + converts once** into one mixed LeRobot dataset.

---

## 7) Open-loop eval (check **each** skill)

Pick trajectory IDs that belong to different skills (inspect episode order in staging / dataset meta).

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "${CKPT}" \
  --traj-ids 0 1 2 \
  --action-horizon 16 \
  --steps 300
```

- Plots should track GT for **every** skill you care about.
- One skill much worse → more demos / balance / wrong `--task` on those episodes.
- MAE/MSE NaN → bad checkpoint weights; try an earlier `checkpoint-*`.

Details: [`MULTI_TASK_PIPELINE.md`](MULTI_TASK_PIPELINE.md) §6.

---

## 8) Deploy — same checkpoint, change only `--task`

Server (once):

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${CKPT}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path record/custom_3cam_config.py \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

Client — skill switch is **language only**:

```bash
# Task 1
uv run python record/policy_client_3cam.py \
  --host localhost --port 5555 \
  --task "pick up the object and place it in the tray" \
  --robot robstride --use-usb-camera-ports \
  --camera-fps 30 --image-height 640 --image-width 640 \
  --dry-run-robstride --control-mode chunk --rate-hz 30 --max-steps 10

# Task 2 (only --task changes)
uv run python record/policy_client_3cam.py \
  --host localhost --port 5555 \
  --task "pick up the object and place it in the bowl" \
  --robot robstride --use-usb-camera-ports \
  --camera-fps 30 --image-height 640 --image-width 640 \
  --dry-run-robstride --control-mode chunk --rate-hz 30 --max-steps 10
```

After finite targets look good, add `--apply-actions` (same flags as [`Record_Flow_README.md`](Record_Flow_README.md) §8).

---

## 9) Adding task 3+ later (short loop)

1. Set new `SKILL_ID` + `TASK` + `RAW_DIR` → record → visualize.  
2. Re-run **§3 staging** including the new folder.  
3. Re-run **§4 convert** (`--overwrite`), **§5 stats**, **§6 finetune** (new or continued `FT_OUT`).  
4. Open-loop on trajs from **all** skills → deploy with matching `--task`.

---

## 10) Cheat sheet — moving from task 1 → task 2

```bash
# --- only these change for collection ---
export SKILL_ID=pick_place_bowl
export TASK="pick up the object and place it in the bowl"
export RAW_DIR="${SSD}/raw/${SKILL_ID}"

# --- after all skills recorded ---
# STAGE_DIR staging (§3) → convert to REPO_ID=…/multitask_3cam (§4)
# → stats (§5) → FT_OUT=…/gr00t_multitask_3cam (§6)
# → open-loop (§7) → policy client --task "${TASK}" (§8)
```

Keep fixed always: `camera_ports.json`, 30 Hz, `NEW_EMBODIMENT`, `custom_3cam_config.py`.

---

## Related

- [`Record_Flow_README.md`](Record_Flow_README.md) — single-task hardware / convert / train details  
- [`MULTI_TASK_PIPELINE.md`](MULTI_TASK_PIPELINE.md) — why language is the skill switch; open-loop rating  
- [`task_dictionary.md`](task_dictionary.md) — Phase 1–5 skill list and variation plans  
