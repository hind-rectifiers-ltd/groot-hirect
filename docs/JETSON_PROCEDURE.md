# Step 1

In every new terminal, copy paste these two code blocks:
```bash
source .venv/bin/activate
source scripts/activate_orin.sh
```

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

# Step 2

**Before every record or inference session** (or after moving USB cables), confirm labels with a live preview:

```bash
python record/preview_three_cameras.py
```

**If This doesn't work then use below steps for debugging**

Wave each physical camera; the **HEAD**, **LEFT WRIST**, and **RIGHT WRIST** banners must match the view. Press **Esc** to quit.

If a cable moved or you are setting up a new machine:

```bash
# Discover instance= all cameras at a time
python record/preview_three_cameras.py --preview-cameras
# Discover instance= ids one camera at a time (Space = next, Esc = quit)
python record/preview_three_cameras.py --preview-all-cameras

# Text-only listing
python record/record_episodes_3cam.py --list-cameras-working
```

# Step 3

### Read Follower Joints
```bash
python scripts/read_follower_joints.py --format table
```

# Steps 4

### Run Data Collection Code
```bash
python record/record_episodes_3cam.py \
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
**NOTE: Change --task flag depending on the task to be performed**