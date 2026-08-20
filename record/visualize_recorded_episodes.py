#!/usr/bin/env python3
"""
Visualize recorded HDF5 episodes from record/record_episodes_3cam.py.

Inspired by the Hugging Face LeRobot dataset visualizer (multi-camera strip + playback).
Shows cam_head | cam_left_wrist | cam_right_wrist with timestep and task overlay, plus
**time-series plots** of full-episode **qpos** and **action** (all joints) with a **yellow cursor** at the current frame.

Uses OpenCV windows when HighGUI is available; otherwise falls back to Matplotlib (works with
`opencv-python-headless`). Override with `--gui opencv` or `--gui matplotlib`. Use `--save-video`
for a file-only export without any display.

Usage:
  uv run python record/visualize_recorded_episodes.py --data-dir ./record/cube_pick_place

  # All sessions under ./record (recursive search for episode_*.hdf5):
  uv run python record/visualize_recorded_episodes.py --data-dir ./record --recursive

  # Export preview without display:
  uv run python record/visualize_recorded_episodes.py --data-dir ./record/foo \\
    --episode 0 --save-video ./preview.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Must match record/record_episodes_3cam.py
CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")

CAMERA_DISPLAY_LABELS: dict[str, str] = {
    "cam_head": "HEAD",
    "cam_left_wrist": "LEFT WRIST",
    "cam_right_wrist": "RIGHT WRIST",
}

# Banner colors (RGB) — same roles as record/usb_cameras.py preview
CAMERA_BANNER_RGB: dict[str, tuple[int, int, int]] = {
    "cam_head": (40, 160, 40),
    "cam_left_wrist": (30, 100, 180),
    "cam_right_wrist": (200, 90, 30),
}

# Default humanoid 16-DoF naming (7+1 per arm); also recognize legacy 12-DoF labels
_DEFAULT_JOINT_LABELS_16 = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "Lg",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "Rg",
)
_DEFAULT_JOINT_LABELS_12 = (
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "Lg",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "Rg",
)


def _joint_labels(dim: int) -> list[str]:
    if dim == len(_DEFAULT_JOINT_LABELS_16):
        return list(_DEFAULT_JOINT_LABELS_16)
    if dim == len(_DEFAULT_JOINT_LABELS_12):
        return list(_DEFAULT_JOINT_LABELS_12)
    return [f"j{i}" for i in range(dim)]


def _as_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def plot_qpos_action_on_axes(
    ax_q,
    ax_a,
    qpos: np.ndarray,
    action: np.ndarray,
    frame_i: int,
) -> None:
    """Draw full-episode qpos / action vs frame index with a vertical cursor at frame_i."""
    import matplotlib.pyplot as plt

    qpos = _as_2d(np.asarray(qpos, dtype=np.float64))
    action = _as_2d(np.asarray(action, dtype=np.float64))
    T = min(len(qpos), len(action))
    if T == 0:
        ax_q.clear()
        ax_a.clear()
        ax_q.text(0.5, 0.5, "no frames", ha="center", va="center", transform=ax_q.transAxes)
        return
    qpos = qpos[:T]
    action = action[:T]
    D = min(qpos.shape[1], action.shape[1])
    labels = _joint_labels(D)
    x = np.arange(T)
    # Use fixed categorical tab20 colors (not interpolated samples) so
    # joint->color mapping is stable and visually identical across subplots.
    palette = list(plt.get_cmap("tab20").colors)

    ax_q.clear()
    ax_a.clear()
    for i in range(D):
        color = palette[i % len(palette)]
        lbl = labels[i] if D <= 16 else None
        ax_q.plot(x, qpos[:, i], color=color, lw=1.0, label=lbl)
        ax_a.plot(x, action[:, i], color=color, lw=1.0, label=lbl)

    # Keep identical y-scale on both axes so same-joint traces are directly comparable.
    y_all = np.concatenate([qpos[:, :D].reshape(-1), action[:, :D].reshape(-1)], axis=0)
    finite = np.isfinite(y_all)
    if np.any(finite):
        y_min = float(np.min(y_all[finite]))
        y_max = float(np.max(y_all[finite]))
        if y_max <= y_min:
            pad = 0.5
        else:
            pad = 0.05 * (y_max - y_min)
        y_lo = y_min - pad
        y_hi = y_max + pad
        ax_q.set_ylim(y_lo, y_hi)
        ax_a.set_ylim(y_lo, y_hi)
    fi = int(np.clip(frame_i, 0, T - 1))
    ax_q.axvline(fi, color="yellow", lw=2.0, zorder=10)
    ax_a.axvline(fi, color="yellow", lw=2.0, zorder=10)
    ax_q.set_ylabel("qpos")
    ax_q.set_title("Joint positions (observation.state)")
    ax_q.grid(True, alpha=0.3)
    if D <= 16:
        ax_q.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.7)
    ax_a.set_ylabel("action")
    ax_a.set_xlabel("frame index")
    ax_a.set_title("Actions (commands)")
    ax_a.grid(True, alpha=0.3)
    if D <= 16:
        ax_a.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.7)


def render_trajectory_panel_bgr(
    qpos: np.ndarray,
    action: np.ndarray,
    frame_i: int,
    width_px: int,
    height_px: int = 300,
) -> np.ndarray:
    """Rasterize qpos/action trajectory plots for stacking under OpenCV (Agg, no GUI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    qpos = _as_2d(np.asarray(qpos, dtype=np.float64))
    action = _as_2d(np.asarray(action, dtype=np.float64))
    T = min(len(qpos), len(action))
    if T == 0:
        import cv2

        return np.zeros((height_px, width_px, 3), dtype=np.uint8)

    w_in = width_px / 100.0
    h_in = height_px / 100.0
    fig, (ax_q, ax_a) = plt.subplots(2, 1, figsize=(w_in, h_in), dpi=100, sharex=True)
    plot_qpos_action_on_axes(ax_q, ax_a, qpos, action, frame_i)
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = np.asarray(buf[:, :, :3])
    plt.close(fig)
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != width_px:
        nh = max(180, int(bgr.shape[0] * width_px / bgr.shape[1]))
        bgr = cv2.resize(bgr, (width_px, nh), interpolation=cv2.INTER_AREA)
    return bgr


def combine_strip_and_plots_bgr(
    strip_bgr: np.ndarray, qpos: np.ndarray, action: np.ndarray, frame_i: int
) -> np.ndarray:
    """Append trajectory-plot panel below the camera strip (BGR)."""
    import cv2

    w = strip_bgr.shape[1]
    plot_bgr = render_trajectory_panel_bgr(qpos, action, frame_i, width_px=w)
    if plot_bgr.shape[1] != w:
        plot_bgr = cv2.resize(
            plot_bgr, (w, max(200, int(plot_bgr.shape[0] * w / plot_bgr.shape[1]))), interpolation=cv2.INTER_AREA
        )
    return np.vstack([strip_bgr, plot_bgr])


def _decode_task(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw) if raw is not None else ""


def discover_episodes(data_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(data_dir.rglob("episode_*.hdf5"))
    else:
        files = sorted(data_dir.glob("episode_*.hdf5"))
    return files


def infer_fps(f: h5py.File) -> float:
    fps_attr = f.attrs.get("fps", None)
    if fps_attr is not None:
        try:
            v = float(fps_attr)
            if v > 0.5:
                return v
        except (TypeError, ValueError):
            pass
    if "/timestamp" in f:
        ts = np.asarray(f["/timestamp"][:], dtype=np.float64)
        if len(ts) > 1:
            d = np.diff(ts)
            d = d[d > 1e-6]
            if len(d) > 0:
                return float(1.0 / np.median(d))
    return 30.0


def load_episode(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        task = _decode_task(f.attrs.get("task", ""))
        fps = infer_fps(f)
        qpos = np.asarray(f["/observations/qpos"][:], dtype=np.float32)
        action = np.asarray(f["/action"][:], dtype=np.float32)
        t = len(qpos)
        imgs = {}
        for cam in CAMERA_NAMES:
            key = f"/observations/images/{cam}"
            if key not in f:
                raise KeyError(f"{path}: missing {key}")
            imgs[cam] = np.asarray(f[key][:], dtype=np.uint8)
            if len(imgs[cam]) != t:
                t = min(t, len(imgs[cam]))
        qpos = qpos[:t]
        action = action[: min(len(action), t)]
        for cam in CAMERA_NAMES:
            imgs[cam] = imgs[cam][:t]
    return {"task": task, "fps": fps, "qpos": qpos, "action": action, "images": imgs, "length": t}


def _annotate_camera_tile_rgb(tile: np.ndarray, cam: str) -> np.ndarray:
    """Draw a colored HEAD / LEFT WRIST / RIGHT WRIST banner on one camera tile."""
    import cv2

    label = CAMERA_DISPLAY_LABELS.get(cam, cam)
    banner_rgb = CAMERA_BANNER_RGB.get(cam, (80, 80, 80))
    out = np.asarray(tile, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 10)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    banner_bgr = (banner_rgb[2], banner_rgb[1], banner_rgb[0])
    cv2.rectangle(bgr, (0, 0), (w, bar_h), banner_bgr, thickness=-1)
    cv2.putText(
        bgr,
        label,
        (8, bar_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_rgb_strip_tiles(
    images: dict[str, np.ndarray],
    frame_idx: int,
    target_h: int = 240,
) -> np.ndarray:
    """RGB horizontal strip (H, W, 3). Resize only; works with headless OpenCV."""
    import cv2

    tiles = []
    for cam in CAMERA_NAMES:
        img = images[cam][frame_idx]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Bad image shape for {cam}: {img.shape}")
        h, w = img.shape[:2]
        new_w = int(w * (target_h / h))
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        tiles.append(_annotate_camera_tile_rgb(resized, cam))
    return np.concatenate(tiles, axis=1)


def opencv_gui_available() -> bool:
    try:
        import cv2

        cv2.namedWindow("__viz_gui_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__viz_gui_test__")
        return True
    except Exception:
        return False


def _format_qpos_lines(qpos_row: np.ndarray, precision: int = 4) -> list[str]:
    """
    Format a qpos vector as one-or-more compact lines for on-image overlay.

    - 16-D vectors are split into left arm and right arm rows using the standard labels.
    - Other dimensions are split into chunks of up to 6 joints per line.
    - Decimal output (no scientific notation), with a leading sign so columns line up.
    """
    arr = np.asarray(qpos_row, dtype=np.float64).reshape(-1)
    dim = arr.size
    labels = _joint_labels(dim)
    chunk = 6  # joints per row

    def fmt(label: str, v: float) -> str:
        return f"{label}={v:+.{precision}f}"

    lines: list[str] = []
    for start in range(0, dim, chunk):
        end = min(start + chunk, dim)
        row = "  ".join(fmt(labels[i], float(arr[i])) for i in range(start, end))
        lines.append(row)
    return lines


def build_frame_strip(
    images: dict[str, np.ndarray],
    frame_idx: int,
    episode_label: str,
    task: str,
    qpos_row: np.ndarray | None,
    target_h: int = 240,
) -> np.ndarray:
    """BGR strip with text overlay for OpenCV imshow / VideoWriter."""
    strip = build_rgb_strip_tiles(images, frame_idx, target_h=target_h)
    try:
        import cv2

        out = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
        y = 28
        cv2.putText(
            out,
            f"{episode_label}  frame {frame_idx}",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y += 26
        task_show = task.replace("\n", " ")[:120]
        cv2.putText(out, task_show, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if qpos_row is not None and len(qpos_row) > 0:
            y += 24
            cv2.putText(out, "qpos:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            line_h = 20
            for line in _format_qpos_lines(qpos_row):
                y += line_h
                cv2.putText(
                    out,
                    line,
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )
        return out
    except ImportError:
        return strip


def run_viewer_matplotlib(
    episodes: list[Path],
    start_episode: int,
    playback_fps: float | None,
) -> None:
    """Interactive viewer when OpenCV has no HighGUI (e.g. opencv-python-headless)."""
    import matplotlib.pyplot as plt

    ep_i = max(0, min(start_episode, len(episodes) - 1))
    data = load_episode(episodes[ep_i])
    fps = playback_fps if playback_fps is not None else data["fps"]

    state: dict = {
        "ep_i": ep_i,
        "frame_i": 0,
        "paused": False,
        "data": data,
        "fps": fps,
    }

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.8, 1.15, 1.15], hspace=0.38)
    ax_img = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1])
    ax_a = fig.add_subplot(gs[2])
    try:
        mgr = fig.canvas.manager
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("Recorded dataset — cameras + qpos / action trajectories")
    except Exception:
        pass
    plt.subplots_adjust(top=0.94, left=0.06, right=0.98)

    def load_ep(ep_index: int) -> None:
        state["data"] = load_episode(episodes[ep_index])
        state["fps"] = playback_fps if playback_fps is not None else state["data"]["fps"]
        state["frame_i"] = 0

    def redraw() -> None:
        d = state["data"]
        ei = state["ep_i"]
        fi = state["frame_i"] % max(d["length"], 1)
        rgb = build_rgb_strip_tiles(d["images"], fi)
        ax_img.clear()
        ax_img.imshow(rgb)
        ax_img.axis("off")
        title_lines = [
            f"{episodes[ei].stem}  frame {fi}/{max(d['length'] - 1, 0)}",
            d["task"][:100],
        ]
        if len(d["qpos"]) > fi:
            title_lines.append("qpos: " + "  |  ".join(_format_qpos_lines(d["qpos"][fi])))
        ax_img.set_title("\n".join(title_lines), fontsize=9, loc="left")
        if len(d["qpos"]) and len(d["action"]):
            plot_qpos_action_on_axes(ax_q, ax_a, d["qpos"], d["action"], fi)
        else:
            ax_q.clear()
            ax_a.clear()
            msg = "no qpos/action"
            ax_q.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_q.transAxes)
            ax_a.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_a.transAxes)

    def on_key(event) -> None:
        if event.key is None:
            return
        key = event.key
        d = state["data"]
        n_frames = d["length"]
        if key in ("q", "escape"):
            plt.close(fig)
            return
        if key == " ":
            state["paused"] = not state["paused"]
            return
        if key == "n":
            state["ep_i"] = (state["ep_i"] + 1) % len(episodes)
            load_ep(state["ep_i"])
            redraw()
            fig.canvas.draw_idle()
            return
        if key == "p":
            state["ep_i"] = (state["ep_i"] - 1) % len(episodes)
            load_ep(state["ep_i"])
            redraw()
            fig.canvas.draw_idle()
            return
        if key == "r":
            state["frame_i"] = 0
            redraw()
            fig.canvas.draw_idle()
            return
        if key == ",":
            state["paused"] = True
            state["frame_i"] = max(0, state["frame_i"] - 1)
            redraw()
            fig.canvas.draw_idle()
            return
        if key == ".":
            state["paused"] = True
            state["frame_i"] = min(n_frames - 1, state["frame_i"] + 1)
            redraw()
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: Space pause | n/p episode | , . step | r restart | q quit")
    print("Tip: `uv pip install opencv-python` for native OpenCV windows instead of this viewer.")

    plt.ion()
    try:
        while plt.fignum_exists(fig.number):
            redraw()
            delay = 0.05 if state["paused"] else 1.0 / max(state["fps"], 1e-3)
            plt.pause(delay)
            d = state["data"]
            n_frames = d["length"]
            if n_frames == 0:
                continue
            if not state["paused"]:
                state["frame_i"] = (state["frame_i"] + 1) % n_frames
    finally:
        plt.ioff()
        plt.close("all")


def run_viewer_opencv(
    episodes: list[Path],
    start_episode: int,
    playback_fps: float | None,
) -> None:
    import cv2

    ep_i = max(0, min(start_episode, len(episodes) - 1))
    data = load_episode(episodes[ep_i])
    fps = playback_fps if playback_fps is not None else data["fps"]
    delay_ms = max(1, int(1000 / max(fps, 1e-3)))
    paused = False
    frame_i = 0

    win = "Recorded dataset (LeRobot-style) — Space pause | n/p episode | , . step | q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("Controls:")
    print("  Space     — pause / resume")
    print("  n / p     — next / previous episode")
    print("  , / .     — step one frame back / forward (when paused)")
    print("  r         — restart episode")
    print("  q / ESC   — quit")

    while True:
        n_frames = data["length"]
        if n_frames == 0:
            print("Episode has 0 frames, skipping.")
            ep_i = (ep_i + 1) % len(episodes)
            data = load_episode(episodes[ep_i])
            frame_i = 0
            continue

        frame_i = frame_i % n_frames
        qrow = data["qpos"][frame_i] if len(data["qpos"]) > frame_i else None
        strip = build_frame_strip(data["images"], frame_i, episodes[ep_i].stem, data["task"], qrow)
        combined = combine_strip_and_plots_bgr(strip, data["qpos"], data["action"], frame_i)

        cv2.imshow(win, combined)
        raw_key = cv2.waitKey(delay_ms if not paused else 30)
        key = raw_key & 0xFF if raw_key >= 0 else 0
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            ep_i = (ep_i + 1) % len(episodes)
            data = load_episode(episodes[ep_i])
            fps = playback_fps if playback_fps is not None else data["fps"]
            delay_ms = max(1, int(1000 / max(fps, 1e-3)))
            frame_i = 0
            paused = False
        elif key == ord("p"):
            ep_i = (ep_i - 1) % len(episodes)
            data = load_episode(episodes[ep_i])
            fps = playback_fps if playback_fps is not None else data["fps"]
            delay_ms = max(1, int(1000 / max(fps, 1e-3)))
            frame_i = 0
            paused = False
        elif key == ord("r"):
            frame_i = 0
        elif key == ord(","):
            frame_i = max(0, frame_i - 1)
            paused = True
        elif key == ord("."):
            frame_i = min(n_frames - 1, frame_i + 1)
            paused = True

        if not paused:
            frame_i += 1
            if frame_i >= n_frames:
                frame_i = 0

    cv2.destroyAllWindows()


def run_viewer(
    episodes: list[Path],
    start_episode: int,
    playback_fps: float | None,
    save_video: Path | None,
    gui: str = "auto",
) -> None:
    if not episodes:
        print("No episode_*.hdf5 files found.", file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("Install opencv for visualization: uv pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    ep_i = max(0, min(start_episode, len(episodes) - 1))
    data = load_episode(episodes[ep_i])
    fps = playback_fps if playback_fps is not None else data["fps"]
    if save_video is not None:
        q0 = data["qpos"][0] if len(data["qpos"]) else None
        strip0 = build_frame_strip(
            data["images"],
            0,
            episodes[ep_i].stem,
            data["task"],
            q0,
        )
        frame0 = combine_strip_and_plots_bgr(strip0, data["qpos"], data["action"], 0)
        h, w = frame0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_video), fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"Could not open video writer for {save_video}", file=sys.stderr)
            sys.exit(1)
        for fi in range(data["length"]):
            qrow = data["qpos"][fi] if len(data["qpos"]) > fi else None
            strip = build_frame_strip(
                data["images"], fi, episodes[ep_i].stem, data["task"], qrow
            )
            fr = combine_strip_and_plots_bgr(strip, data["qpos"], data["action"], fi)
            writer.write(fr)
        writer.release()
        print(f"Wrote {save_video} ({data['length']} frames @ {fps:.1f} fps)")
        return

    if gui == "matplotlib":
        try:
            run_viewer_matplotlib(episodes, start_episode, playback_fps)
        except ImportError:
            print(
                "Matplotlib is required for --gui matplotlib. Example: uv pip install matplotlib",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    if gui == "opencv":
        if not opencv_gui_available():
            print(
                "OpenCV was built without GUI support (common with opencv-python-headless). "
                "Install `opencv-python` or run with `--gui auto` / `--gui matplotlib`.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_viewer_opencv(episodes, start_episode, playback_fps)
        return

    # auto
    if opencv_gui_available():
        run_viewer_opencv(episodes, start_episode, playback_fps)
    else:
        print(
            "OpenCV HighGUI not available (typical for opencv-python-headless). "
            "Using Matplotlib window.",
            flush=True,
        )
        try:
            run_viewer_matplotlib(episodes, start_episode, playback_fps)
        except ImportError:
            print(
                "Install matplotlib for the fallback viewer: uv pip install matplotlib\n"
                "Or install OpenCV with GUI: uv pip install opencv-python\n"
                "Or export video only: --save-video out.mp4",
                file=sys.stderr,
            )
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 3-camera HDF5 episodes (record/record_episodes_3cam.py output)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing episode_*.hdf5 (e.g. ./record/cube_pick_place), or parent if --recursive",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find all episode_*.hdf5 under data-dir (multiple sessions).",
    )
    parser.add_argument("--episode", type=int, default=0, help="Start at episode index in sorted list (0-based).")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS (default: from file timestamps or attrs).",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Write concatenated camera video to this path and exit (no GUI).",
    )
    parser.add_argument(
        "--gui",
        choices=("auto", "opencv", "matplotlib"),
        default="auto",
        help=(
            "Display backend: auto (OpenCV window if available, else Matplotlib), "
            "opencv (fail if no HighGUI), matplotlib (always use Matplotlib)."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    episodes = discover_episodes(data_dir, args.recursive)
    run_viewer(episodes, args.episode, args.fps, args.save_video, gui=args.gui)


if __name__ == "__main__":
    main()
