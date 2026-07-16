# Task Dictionary — General Human Skills (Starter Set)

Canonical **language labels**, collection targets, and variation plans for teaching a tabletop / desk humanoid (dual-arm, 3-camera setup) multiple everyday skills with GR00T.

Use with [`MULTI_TASK_PIPELINE.md`](MULTI_TASK_PIPELINE.md) and [`Record_Flow_README.md`](Record_Flow_README.md) / [`record/README.md`](../record/README.md).

---

## How to use this dictionary

1. Pick a **phase** (start with Phase 1).
2. For each skill, copy the **canonical `--task` string exactly** into recording, conversion/`tasks.jsonl`, and inference.
3. Collect the **target episodes** using the **variation plan** (spread demos across setups; do not record 80 identical takes).
4. Keep hardware fixed: cameras (`camera_ports.json`), 30 Hz, same joint order, same embodiment.

### Episode definition

One episode = **one successful, complete skill** from a natural start pose to a clear end state (object placed, drawer open, button pressed, etc.). Drop failures.

### Variation math

```text
target_episodes ≈ num_variation_setups × episodes_per_setup
```

Example: **8 setups × 10 episodes = 80 episodes**.

Prefer more setups with fewer repeats over one setup repeated forever.

### Global collection defaults (unless a skill overrides)

| Item | Default |
|---|---|
| Control / camera rate | 30 Hz |
| Min good episodes / skill (narrow) | ~40–50 |
| Recommended / skill (starter multi-task) | **60–100** |
| Stronger / skill (harder contact skills) | **100–150** |
| Failures kept? | No — redo or discard |
| Idle head/tail | Trim when converting if possible |
| Balance across skills | Keep episode counts within ~1.5× of each other |

### Variation axes (reuse across skills)

| Axis | Examples |
|---|---|
| **Object pose** | XY on table, yaw rotation, near / mid / far reach |
| **Object instance** | 2–4 similar objects (size/color) if the skill is generic |
| **Container / target pose** | Tray/bowl/drawer position shifted |
| **Start pose** | Small arm start variation (not extreme) |
| **Lighting** | Normal + slightly darker / brighter |
| **Distractors** | 0–2 unrelated objects in view (optional Phase 2+) |
| **Approach** | Left-biased vs right-biased reach when natural |

---

## Phase roadmap

| Phase | Goal | Skills below | Suggested total demos |
|---|---|---|---|
| **1 — Foundation** | Reach, grasp, place | 1–4 | ~280–360 |
| **2 — Transfer & sort** | Containers, stacking, sorting | 5–8 | ~280–360 |
| **3 — Articulated** | Buttons, drawers, doors, lids | 9–12 | ~320–400 |
| **4 — Bimanual / social** | Two-hand and handover | 13–15 | ~240–300 |
| **5 — Short routines** | Chain Phase 1–4 skills (optional later) | 16–18 | after atomics work |

Do **not** start Phase 5 until open-loop + live demos look solid on Phases 1–2.

---

## Phase 1 — Foundation (pick / place family)

### 1. Pick and place into tray

| Field | Detail |
|---|---|
| **Skill ID** | `pick_place_tray` |
| **Canonical task** | `pick up the object and place it in the tray` |
| **Difficulty** | Easy–medium (your current baseline) |
| **Success** | Object ends fully in tray; arms settle; no drop outside tray |
| **Target episodes** | **80** |
| **Variation setups** | **8** |
| **Episodes / setup** | **10** |
| **What to vary** | Object XY + yaw (4 table regions × 2 yaw); tray left/center/right (fold into the 8 setups); 2 object instances; normal lighting |
| **Keep fixed** | Same tray type; same table height; same cameras |
| **Notes** | Match the sentence you already use end-to-end. Good first multi-task anchor. |

### 2. Pick and place into bowl / bin

| Field | Detail |
|---|---|
| **Skill ID** | `pick_place_bowl` |
| **Canonical task** | `pick up the object and place it in the bowl` |
| **Difficulty** | Easy–medium |
| **Success** | Object inside bowl/bin; no rim hang |
| **Target episodes** | **70** |
| **Variation setups** | **7** |
| **Episodes / setup** | **10** |
| **What to vary** | Object pose; bowl position; bowl slightly larger/smaller if available; approach from left vs right |
| **Keep fixed** | One clear “bowl” concept so language stays unambiguous |
| **Notes** | Teaches a second place-target without changing grasp much. |

### 3. Move object to a marked spot

| Field | Detail |
|---|---|
| **Skill ID** | `place_on_marker` |
| **Canonical task** | `pick up the object and place it on the marker` |
| **Difficulty** | Medium (precision) |
| **Success** | Object resting on/near marker (define tolerance, e.g. within marker pad) |
| **Target episodes** | **80** |
| **Variation setups** | **8** |
| **Episodes / setup** | **10** |
| **What to vary** | Marker position; object start far from marker; object size; slight lighting change |
| **Keep fixed** | High-contrast marker visible in head cam |
| **Notes** | Builds precise place; useful before stacking. |

### 4. Hand object from left to right workspace

| Field | Detail |
|---|---|
| **Skill ID** | `relocate_across_table` |
| **Canonical task** | `pick up the object and move it to the other side of the table` |
| **Difficulty** | Easy–medium |
| **Success** | Object clearly on the opposite half of the table |
| **Target episodes** | **60** |
| **Variation setups** | **6** |
| **Episodes / setup** | **10** |
| **What to vary** | Start side (left→right and right→left as separate setups); object type; mid-table obstacles none at first |
| **Keep fixed** | Clear left/right halves |
| **Notes** | Long reach / transport without a container. |

---

## Phase 2 — Transfer, stack, sort

### 5. Put object into drawer (drawer already open)

| Field | Detail |
|---|---|
| **Skill ID** | `place_in_open_drawer` |
| **Canonical task** | `pick up the object and place it in the open drawer` |
| **Difficulty** | Medium |
| **Success** | Object inside drawer cavity; not on lip |
| **Target episodes** | **80** |
| **Variation setups** | **8** |
| **Episodes / setup** | **10** |
| **What to vary** | Object pose; drawer open amount (mostly open vs half); object size |
| **Keep fixed** | Drawer pre-opened by human — do **not** mix open-drawer motion into this skill |
| **Notes** | Separates “place into cavity” from “open drawer”. |

### 6. Stack object onto another object

| Field | Detail |
|---|---|
| **Skill ID** | `stack_objects` |
| **Canonical task** | `stack the object on top of the other object` |
| **Difficulty** | Medium–hard |
| **Success** | Stable stack; upper object does not immediately fall |
| **Target episodes** | **100** |
| **Variation setups** | **10** |
| **Episodes / setup** | **10** |
| **What to vary** | Base object position/yaw; top object; height of base; gentle vs firm place |
| **Keep fixed** | Stackable pair (e.g. cube on cube / cup on block) |
| **Notes** | Needs careful teleop; more demos help. |

### 7. Sort object by type into two bins

| Field | Detail |
|---|---|
| **Skill ID** | `sort_into_bins` |
| **Canonical task** | `sort the object into the correct bin` |
| **Difficulty** | Medium (language + visual class) |
| **Success** | Correct bin for the object class shown |
| **Target episodes** | **90** |
| **Variation setups** | **9** |
| **Episodes / setup** | **10** |
| **What to vary** | 2 classes × several instances; bin positions swapped in some setups; object start pose |
| **Keep fixed** | Exactly two classes and two bins; teach humans the mapping before recording |
| **Notes** | Use **one** sentence; class is implied by the scene. Optional later: two explicit sentences per class. |

### 8. Pour / tip into container (easy prop)

| Field | Detail |
|---|---|
| **Skill ID** | `pour_into_container` |
| **Canonical task** | `pour the contents into the container` |
| **Difficulty** | Hard |
| **Success** | Most contents end in target (define with dry beans/rice, not liquids first) |
| **Target episodes** | **100** |
| **Variation setups** | **10** |
| **Episodes / setup** | **10** |
| **What to vary** | Cup fill level; container pose; pour left vs right hand if your teleop allows |
| **Keep fixed** | Dry media only until reliable |
| **Notes** | Skip until Phase 1 is strong; contact + tilt are brittle. |

---

## Phase 3 — Articulated & press

### 9. Push button / switch

| Field | Detail |
|---|---|
| **Skill ID** | `push_button` |
| **Canonical task** | `push the button` |
| **Difficulty** | Easy–medium |
| **Success** | Clear press (click / travel); arm retracts slightly |
| **Target episodes** | **60** |
| **Variation setups** | **6** |
| **Episodes / setup** | **10** |
| **What to vary** | Button panel pose; height; which finger/side of gripper; mild lighting |
| **Keep fixed** | One large easy button first |
| **Notes** | Short episodes; good for language conditioning checks. |

### 10. Open drawer

| Field | Detail |
|---|---|
| **Skill ID** | `open_drawer` |
| **Canonical task** | `open the drawer` |
| **Difficulty** | Medium–hard |
| **Success** | Drawer opens past a marked amount (e.g. >50% travel) |
| **Target episodes** | **100** |
| **Variation setups** | **10** |
| **Episodes / setup** | **10** |
| **What to vary** | Handle approach angle; start distance; drawer closed flush vs slightly ajar; left vs right arm if both can reach |
| **Keep fixed** | Same drawer hardware |
| **Notes** | Keep separate from place-in-drawer. |

### 11. Close drawer

| Field | Detail |
|---|---|
| **Skill ID** | `close_drawer` |
| **Canonical task** | `close the drawer` |
| **Difficulty** | Medium |
| **Success** | Drawer fully closed / latched |
| **Target episodes** | **70** |
| **Variation setups** | **7** |
| **Episodes / setup** | **10** |
| **What to vary** | How far drawer starts open; push vs pull-close style consistent with teleop |
| **Keep fixed** | Same drawer as open skill |
| **Notes** | Pair with open_drawer for symmetric coverage. |

### 12. Open / close lid (box or bin)

| Field | Detail |
|---|---|
| **Skill ID** | `open_lid` |
| **Canonical task** | `open the lid` |
| **Difficulty** | Medium–hard |
| **Success** | Lid open enough to access inside |
| **Target episodes** | **90** |
| **Variation setups** | **9** |
| **Episodes / setup** | **10** |
| **What to vary** | Lid hinge side; box pose; grasp point on lid |
| **Keep fixed** | One lid type; add `close the lid` as a **separate** skill later if needed |
| **Notes** | Add `close_lid` with the same episode/variation pattern when ready. |

---

## Phase 4 — Bimanual / handover

### 13. Two-hand lift of a tray

| Field | Detail |
|---|---|
| **Skill ID** | `bimanual_lift_tray` |
| **Canonical task** | `lift the tray with both hands` |
| **Difficulty** | Hard |
| **Success** | Tray rises cleanly; stays level enough not to spill (empty tray first) |
| **Target episodes** | **100** |
| **Variation setups** | **10** |
| **Episodes / setup** | **10** |
| **What to vary** | Tray position; empty vs light weight; small yaw of tray |
| **Keep fixed** | Handles or clear grasp edges visible to wrist cams |
| **Notes** | Needs coordinated teleop; collect only high-quality demos. |

### 14. Pass object from one arm to the other

| Field | Detail |
|---|---|
| **Skill ID** | `handover_between_arms` |
| **Canonical task** | `pass the object from one hand to the other` |
| **Difficulty** | Hard |
| **Success** | Object ends in the other gripper; no drop |
| **Target episodes** | **100** |
| **Variation setups** | **10** |
| **Episodes / setup** | **10** |
| **What to vary** | Left→right and right→left; object size; handover height |
| **Keep fixed** | Mid-torso handover zone visible in head cam |
| **Notes** | Excellent bimanual primitive; high demo quality required. |

### 15. Hand object to a person (reachable zone)

| Field | Detail |
|---|---|
| **Skill ID** | `handover_to_person` |
| **Canonical task** | `hand the object to the person` |
| **Difficulty** | Hard |
| **Success** | Object held out in a clear handover pose; person can take it (or soft place into hand) |
| **Target episodes** | **80** |
| **Variation setups** | **8** |
| **Episodes / setup** | **10** |
| **What to vary** | Person standing left/center/right; object type; height of hold-out |
| **Keep fixed** | Safe slow motion; same “person” visual cue (e.g. hand in frame) |
| **Notes** | Safety first; optional until bimanual basics work. |

---

## Phase 5 — Short routines (only after atomics work)

These are **optional**. Prefer chaining Phase 1–4 skills at deploy time with separate `--task` calls before collecting long routines.

### 16. Clear object into tray then push tray forward

| Field | Detail |
|---|---|
| **Skill ID** | `clear_and_present_tray` |
| **Canonical task** | `put the object in the tray and push the tray forward` |
| **Difficulty** | Hard (long horizon) |
| **Success** | Object in tray and tray translated forward past a line |
| **Target episodes** | **80** |
| **Variation setups** | **8** |
| **Episodes / setup** | **10** |
| **What to vary** | Object/tray start poses; push distance |
| **Notes** | Only if you need one-shot language for the whole routine. |

### 17. Open drawer, place object, close drawer

| Field | Detail |
|---|---|
| **Skill ID** | `store_object_in_drawer` |
| **Canonical task** | `open the drawer, place the object inside, and close the drawer` |
| **Difficulty** | Hard |
| **Success** | Object stored; drawer closed |
| **Target episodes** | **100** |
| **Variation setups** | **10** |
| **Episodes / setup** | **10** |
| **What to vary** | Object pose; how closed the drawer starts |
| **Notes** | Alternative: keep three atomic skills and sequence them in software. |

### 18. Tidy: pick distractor and place in bin

| Field | Detail |
|---|---|
| **Skill ID** | `tidy_distractor_to_bin` |
| **Canonical task** | `pick up the distractor and place it in the bin` |
| **Difficulty** | Medium |
| **Success** | Named distractor in bin; main task object untouched |
| **Target episodes** | **70** |
| **Variation setups** | **7** |
| **Episodes / setup** | **10** |
| **What to vary** | Which item is the distractor; bin pose; clutter level (1–3 extras) |
| **Notes** | Good “cleanup” skill once pick-place is solid. |

---

## Master summary table

| # | Skill ID | Canonical `--task` | Episodes | Setups × per setup | Phase | Difficulty |
|---|---|---|---|---|---|---|
| 1 | `pick_place_tray` | `pick up the object and place it in the tray` | 80 | 8 × 10 | 1 | Easy–med |
| 2 | `pick_place_bowl` | `pick up the object and place it in the bowl` | 70 | 7 × 10 | 1 | Easy–med |
| 3 | `place_on_marker` | `pick up the object and place it on the marker` | 80 | 8 × 10 | 1 | Medium |
| 4 | `relocate_across_table` | `pick up the object and move it to the other side of the table` | 60 | 6 × 10 | 1 | Easy–med |
| 5 | `place_in_open_drawer` | `pick up the object and place it in the open drawer` | 80 | 8 × 10 | 2 | Medium |
| 6 | `stack_objects` | `stack the object on top of the other object` | 100 | 10 × 10 | 2 | Med–hard |
| 7 | `sort_into_bins` | `sort the object into the correct bin` | 90 | 9 × 10 | 2 | Medium |
| 8 | `pour_into_container` | `pour the contents into the container` | 100 | 10 × 10 | 2 | Hard |
| 9 | `push_button` | `push the button` | 60 | 6 × 10 | 3 | Easy–med |
| 10 | `open_drawer` | `open the drawer` | 100 | 10 × 10 | 3 | Med–hard |
| 11 | `close_drawer` | `close the drawer` | 70 | 7 × 10 | 3 | Medium |
| 12 | `open_lid` | `open the lid` | 90 | 9 × 10 | 3 | Med–hard |
| 13 | `bimanual_lift_tray` | `lift the tray with both hands` | 100 | 10 × 10 | 4 | Hard |
| 14 | `handover_between_arms` | `pass the object from one hand to the other` | 100 | 10 × 10 | 4 | Hard |
| 15 | `handover_to_person` | `hand the object to the person` | 80 | 8 × 10 | 4 | Hard |
| 16 | `clear_and_present_tray` | `put the object in the tray and push the tray forward` | 80 | 8 × 10 | 5 | Hard |
| 17 | `store_object_in_drawer` | `open the drawer, place the object inside, and close the drawer` | 100 | 10 × 10 | 5 | Hard |
| 18 | `tidy_distractor_to_bin` | `pick up the distractor and place it in the bin` | 70 | 7 × 10 | 5 | Medium |

**Phase 1 only (recommended first multi-task run):** ~290 episodes across 4 skills.  
**Phases 1–2:** ~650.  
**All atomics (1–15):** ~1360 (collect gradually; do not block on the full set).

---

## Suggested first multi-task bundle

If you want a practical starting dictionary **this month**:

| Skill ID | Episodes | Why |
|---|---|---|
| `pick_place_tray` | 80 | Already proven in your pipeline |
| `pick_place_bowl` | 70 | Second place target |
| `push_button` | 60 | Short, different verb, good language test |
| `open_drawer` | 100 | Articulated skill, high value |

**Total ≈ 310 demos**, then one multi-task finetune → open-loop per skill → deploy by changing `--task`.

---

## Recording folder layout

```text
record/
  pick_place_tray/
  pick_place_bowl/
  place_on_marker/
  push_button/
  open_drawer/
  ...
```

Each folder: only episodes for that skill, all with the **same** `--task` string from this dictionary.

---

## Phrase rules (do not break)

| Do | Don't |
|---|---|
| Use the exact canonical sentence | Paraphrase between record and deploy |
| One skill ↔ one sentence | One sentence that means different motions |
| Separate `open_drawer` vs `place_in_open_drawer` | Mix open + place in one label while still learning atomics |
| Balance episode counts | 200 of tray + 15 of drawer |

---

## Related docs

- [`MULTI_TASK_PIPELINE.md`](MULTI_TASK_PIPELINE.md) — collect → convert → train → open-loop → deploy
- [`Record_Flow_README.md`](Record_Flow_README.md) / [`record/README.md`](../record/README.md) — commands for this embodiment
