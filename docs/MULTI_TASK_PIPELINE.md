# Multi-Task Pipeline: Collect → Convert → Train → Open-loop → Deploy

Reference guide for teaching **multiple language-conditioned skills** on the same robot embodiment with Isaac GR00T N1.7.

This extends the single-task flow (`pick up the object and place it in the tray`) to several skills, using the same stages you already use for one task.

For day-to-day commands (record, convert, finetune, inference), see [`record/README.md`](record/README.md).

---

## 1. Mental model

GR00T is a **vision–language–action (VLA)** policy. It does not use a hard-coded “mode switch” per skill.

| Input | Role |
|---|---|
| Cameras | What the robot sees |
| Joint state (`qpos`) | Where the arms are |
| **Language task string** | Which skill to perform |
| → Actions | What to command next |

**Single-task** and **multi-task** use the same pipeline. Multi-task means:

1. Record demos under **several canonical task sentences**
2. Convert and train **one** checkpoint on the mixed data
3. Run **open-loop sanity evaluation** before hardware
4. At deploy time, pick the skill by passing the matching `--task` string

Language is the skill switch.

---

## 2. End-to-end flow

```text
Collect demos (per skill)
        ↓
Convert → GR00T LeRobot
        ↓
Train (one multi-task finetune)
        ↓
Open-loop sanity evaluation
        ↓
Deploy (same checkpoint; change --task)
```

| Stage | What you do |
|---|---|
| **Collect** | Teleop demos per skill, fixed task text, same cameras/joints |
| **Convert** | HDF5 → LeRobot; `tasks.jsonl` lists all skills |
| **Train** | Finetune one model on the mixed dataset |
| **Open-loop eval** | Compare predicted vs recorded actions offline (no robot motion) |
| **Deploy** | Run server + policy client; select skill via `--task` |

---

## 3. Define the task dictionary first

Before collecting, freeze a small list of skills with **exact** deploy-time sentences.

| Skill ID | Canonical `--task` string | Episode = |
|---|---|---|
| `pick_place_tray` | `pick up the object and place it in the tray` | One full pick → place |
| `open_drawer` | `open the drawer` | One full open |
| `push_button` | `push the button` | One full press |

### Guidelines

- One episode = **one complete skill**, one sentence.
- Prefer **atomic** skills early; chain them later for long-horizon routines.
- Do **not** casually paraphrase (`pick cube` vs `pick up the object…`) unless you intentionally want language robustness.
- If you would type a different command at deploy time, treat it as a **separate** task string.

Phrase discipline: **record text ≡ `meta/tasks.jsonl` text ≡ inference `--task`**.

Starter skill list with episode counts and variation plans: [`task_dictionary.md`](task_dictionary.md).

---

## 4. Collect data

### Keep fixed across all skills

- Same cameras and USB port mapping (`record/camera_ports.json`)
- Same joint order and control rate (e.g. 30 Hz)
- Same embodiment tag / modality config
- Same roughly similar workspace scale

### What changes per episode

- Scene / objects for that skill
- The `--task` string for that skill

### Suggested folder layout

```text
record/
  pick_place_tray/
    episode_000.hdf5
    ...
  open_drawer/
    ...
  push_button/
    ...
```

Separate folders per skill make balancing and re-recording easier.

### How much data

- Per skill: similar quality bar as single-task (often tens to ~100+ **good** episodes; harder skills need more).
- **Balance** matters: 200 of skill A + 20 of skill B → the policy mostly does A.
- Within a skill, prioritize **variation** (object pose, lighting, mild distractors) over raw count alone.

### Quality bar

Drop or redo episodes that have failed executions, bad camera/joint sync, blur, wrong camera mapping, or aborted mid-skill.

---

## 5. Convert and train

### Convert

1. Convert all skill folders into one (or several) GR00T LeRobot dataset(s).
2. Confirm `meta/tasks.jsonl` lists **all** skills with stable indices.
3. Confirm each episode’s language annotation points at the correct task.

### Train (recommended)

- Finetune **one** multi-task checkpoint on the mixed data.
- Same embodiment / modality config as single-task.
- Watch balance; collect more demos for underrepresented skills if needed.

### Optional: one checkpoint per skill

Only if skills strongly conflict or you need max reliability on one skill. Default for multi-task: **one shared checkpoint**.

---

## 6. Open-loop sanity evaluation

Do this **after training, before real-robot deploy**. It is the main offline check that the checkpoint learned something sensible.

### What “open-loop” means

- The script loads **recorded episodes** (cameras + joints + language + ground-truth actions).
- At intervals (`--action-horizon`), it asks the policy: “given this observation + task text, what action chunk would you output?”
- It compares **predicted actions** to the **ground-truth actions** that were teleoped in that episode.
- The robot does **not** move. There is no closed-loop feedback — hence **open-loop**.

Think of it as: *“If I replay the demo’s eyes and joints, does the model’s action look like the demo’s action?”*

It is a **sanity check**, not a full proof of robot success. A model can look decent open-loop and still fail closed-loop (timing, recovery, contact), but a model that looks **terrible** open-loop is almost never worth putting on hardware yet.

### How to run it

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "${CKPT}" \
  --traj-ids 0 1 2 \
  --action-horizon 16 \
  --steps 300
```

Use trajectories that cover **each skill** you care about (different `traj-ids` that used different task strings).

The script also writes per-trajectory plots (default under `/tmp/open_loop_eval/`), with:

- **gt action** — what was recorded
- **pred action** — what the model predicted
- optional **state joints** when shapes match

### What scores you get

For each trajectory the script logs:

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error — average `\|pred − gt\|` over time and action dims |
| **MSE** | Mean Squared Error — average `(pred − gt)²` (penalizes large spikes harder) |
| **Average MAE / MSE** | Mean of those metrics across the `--traj-ids` you ran |

These are **unnormalized** action errors in the same units as your logged actions (for this setup: joint commands, typically **radians** for arms / gripper encoding as recorded).

**Prefer MAE for intuition.** If average MAE ≈ `0.08`, the typical joint error is about **0.08 rad** (~4.6°) across the evaluated dims/timesteps — not a perfect single-joint guarantee, but a useful scale.

MSE is useful mainly to spot “mostly fine but occasional huge wrong jumps.”

### How to rate the score (practical guide)

There is no universal pass/fail number for every robot. Rate results in this order:

#### 1. Read the plots first (most important)

| Plot look | Rating |
|---|---|
| Pred tracks GT shape and timing; small lag/offset | **Good** — proceed to careful hardware |
| Pred follows trend but overshoots / oscillates | **Marginal** — inspect data / checkpoint; hardware only with low speed |
| Pred is flat, noisy, wrong phase, or opposite direction | **Fail** — do not deploy; fix data/train |

If plots look wrong, ignore a “lucky” low average.

#### 2. Use MAE as a rough numeric gate (joint-space radians)

These are **rules of thumb** for a 16-DoF-style dual-arm (7+1 per side) in radians, comparing checkpoints on the **same** dataset and settings. Retune after you see what a known-good / known-bad run produces on your setup.

| Average MAE (approx.) | How to read it |
|---|---|
| **≲ 0.05** | Strong open-loop match — good candidate for deploy |
| **~0.05 – 0.15** | Acceptable / borderline — check plots per joint; try hardware carefully |
| **~0.15 – 0.30** | Weak — likely jerky or wrong on robot; improve data or train longer |
| **≳ 0.30** | Broken open-loop — wrong mapping, bad checkpoint, or failed learning |

Also compare **MSE vs MAE**:

- Low MAE, high MSE → occasional large spikes (dangerous on hardware)
- Both high → systematically wrong predictions

#### 3. Compare fairly

Only compare scores when these match:

- Same dataset path
- Same `--action-horizon` / `--steps`
- Same embodiment / modality config
- Similar trajectory IDs (or the same set)

Then: **lower MAE/MSE is better**. Use this to pick between `checkpoint-8000` vs `checkpoint-12000`, etc.

#### 4. Multi-task check

Run open-loop on episodes from **each** skill.

| Pattern | Likely issue |
|---|---|
| All skills similar MAE, plots track | Healthy multi-task fit |
| One skill much worse | Too few demos / imbalance / wrong task text on those episodes |
| All skills bad | Camera/joint/stats mismatch or weak training |

### What open-loop does *not* tell you

- Contact-rich success (grasp slip, tray collision)
- Recovery after a miss
- Whether a slightly different object pose still works

After a **pass** on open-loop, still do a careful first live run (low speed, clear workspace), as in [`record/README.md`](record/README.md).

### Pass / fail summary

| Result | Action |
|---|---|
| Plots track + MAE in “good/borderline” band | Deploy with caution |
| Plots OK but MAE high, or MSE spikes | Inspect joints; maybe more data / different ckpt |
| Plots diverge or MAE very high | Stay offline; fix data, conversion, task text, or training |

---

## 7. Deploy

Still **one** policy server / one multi-task checkpoint.

| Wanted behavior | Inference |
|---|---|
| Pick-place | `--task "pick up the object and place it in the tray"` |
| Open drawer | `--task "open the drawer"` |
| … | matching canonical string |

A UI can map buttons (“Skill 1”) → fixed strings; internally it remains language conditioning.

Before live runs:

- [ ] Open-loop plots look reasonable for each skill
- [ ] Task strings match `tasks.jsonl` exactly
- [ ] Camera ports match training
- [ ] Embodiment tag / modality config match training
- [ ] Safe ramp / speed limits for the first trials

---

## 8. Recommended path (checklist)

1. **Freeze** the task dictionary (exact sentences).
2. **Collect** per skill in separate folders; same hardware pipeline.
3. **Curate** (drop failures / bad sync).
4. **Convert** to LeRobot; verify `tasks.jsonl` has all skills.
5. **Finetune** one multi-task model.
6. **Open-loop sanity eval** on a few trajectories per skill; rate plots + MAE/MSE.
7. **Deploy** by swapping only `--task`; keep cameras/embodiment fixed.
8. Later: longer horizons or paraphrases — after atomic skills work.

---

## 9. Common failure modes

| Mistake | Symptom |
|---|---|
| Different wording at train vs deploy | Wrong or ignored skill |
| Unbalanced demo counts | Dominant skill “eats” the policy |
| Failed/idle junk in train | Jerky or hesitant behavior |
| Camera mapping drift between skills | Systematic wrong-view failures |
| One vague label for many skills | No reliable language conditioning |
| Skipping open-loop before robot | Burn hours debugging data bugs on hardware |
| Trusting MAE without looking at plots | Miss large per-joint failures averaged away |

---

## 10. Single-task vs multi-task (quick contrast)

| | Single-task | Multi-task |
|---|---|---|
| Task strings | One | Several, frozen dictionary |
| Dataset | One skill | Mixed skills, balanced |
| Checkpoints | One | Usually still one (language-conditioned) |
| Open-loop | A few trajs of that skill | A few trajs **per** skill |
| Deploy | Fixed `--task` | Same model; change `--task` |

---

## 11. Related docs in this repo

- [`Multi_record_flow_guide.md`](Multi_record_flow_guide.md) — **ordered commands**: per-skill folders → stage → convert → multi-task finetune → deploy
- [`task_dictionary.md`](task_dictionary.md) — starter multi-task skill list, episode counts, variation plans
- [`Record_Flow_README.md`](Record_Flow_README.md) / [`record/README.md`](../record/README.md) — record, convert, finetune, open-loop, inference
- [`getting_started/data_preparation.md`](../getting_started/data_preparation.md) — LeRobot / `tasks.jsonl` schema
- [`getting_started/finetune_new_embodiment.md`](../getting_started/finetune_new_embodiment.md) — finetune setup
- [`getting_started/real_world_deployment.md`](../getting_started/real_world_deployment.md) — data quality and deployment guidance
