"""
Stable USB camera selection for the 3-cam GR00T rig (Linux V4L2).

Pins each logical camera role (head / left wrist / right wrist) to a physical USB port
via udev ``ID_PATH_TAG``, so the correct ``/dev/video*`` node is chosen even when numeric
indices change across reboots.

Port mapping is stored in ``record/camera_ports.json`` (edit after
``python record/record_episodes_3cam.py --list-cameras``).

Derived from ``multi_img_client.py`` in the repo root.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Logical names used by record_episodes_3cam / policy_client_3cam.
THREE_CAM_ROLES = ("cam_head", "cam_left_wrist", "cam_right_wrist")

ROLE_DISPLAY_LABELS: dict[str, str] = {
    "cam_head": "HEAD",
    "cam_left_wrist": "LEFT WRIST",
    "cam_right_wrist": "RIGHT WRIST",
}

_DEFAULT_PORTS_FILE = Path(__file__).resolve().parent / "camera_ports.json"

WARMUP_READS = 15

_UDEV_USB_CACHE: dict[str, dict[str, str]] = {}


def default_ports_config_path() -> Path:
    return _DEFAULT_PORTS_FILE


def load_role_usb_select(config_path: Path | None = None) -> dict[str, dict[str, str] | str]:
    """Load role -> USB match spec from JSON; falls back to built-in empty (caller must handle)."""
    path = config_path or _DEFAULT_PORTS_FILE
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): v for k, v in raw.items()}
    return {}


def list_v4l2_device_nodes() -> list[str]:
    paths = glob.glob("/dev/video[0-9]*")

    def sort_key(p: str) -> int:
        m = re.search(r"(\d+)$", p)
        return int(m.group(1)) if m else 0

    return sorted(paths, key=sort_key)


def get_v4l2_physical_device_key(dev_path: str) -> str | None:
    m = re.search(r"video(\d+)$", dev_path)
    if not m:
        return None
    link = f"/sys/class/video4linux/video{m.group(1)}/device"
    try:
        if os.path.exists(link):
            return os.path.realpath(link)
    except OSError:
        pass
    return None


def _read_sysfs_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def find_usb_device_sysfs_dir(device_realpath: str) -> str | None:
    d = device_realpath
    while d and d != "/":
        vid_path = os.path.join(d, "idVendor")
        pid_path = os.path.join(d, "idProduct")
        if os.path.isfile(vid_path) and os.path.isfile(pid_path):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _udev_usb_properties(usb_dir: str) -> dict[str, str]:
    rp = os.path.realpath(usb_dir)
    if rp in _UDEV_USB_CACHE:
        return _UDEV_USB_CACHE[rp]
    props: dict[str, str] = {}
    try:
        completed = subprocess.run(
            ["udevadm", "info", "-q", "property", "-p", rp],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout:
            for line in completed.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    props[k.strip()] = v.strip()
    except (FileNotFoundError, OSError):
        pass
    _UDEV_USB_CACHE[rp] = props
    return props


def get_camera_identity(dev_path: str | int) -> dict[str, str]:
    if not isinstance(dev_path, str) or not sys.platform.startswith("linux"):
        return {}
    phys = get_v4l2_physical_device_key(dev_path)
    out: dict[str, str] = {
        "phys": phys or "",
        "vendor": "",
        "product": "",
        "serial": "",
        "manufacturer": "",
        "product_name": "",
        "usb_port_id": "",
        "kernel_usb_devpath": "",
        "id_path": "",
        "id_path_tag": "",
    }
    if not phys:
        return out

    usb_dir = find_usb_device_sysfs_dir(phys)
    if not usb_dir:
        return out

    out["vendor"] = _read_sysfs_text(os.path.join(usb_dir, "idVendor")).lower()
    out["product"] = _read_sysfs_text(os.path.join(usb_dir, "idProduct")).lower()
    out["serial"] = _read_sysfs_text(os.path.join(usb_dir, "serial"))
    out["manufacturer"] = _read_sysfs_text(os.path.join(usb_dir, "manufacturer"))
    out["product_name"] = _read_sysfs_text(os.path.join(usb_dir, "product"))
    out["usb_port_id"] = os.path.basename(usb_dir)
    rp_usb = os.path.realpath(usb_dir)
    if rp_usb.startswith("/sys"):
        out["kernel_usb_devpath"] = rp_usb[len("/sys") :]

    props = _udev_usb_properties(usb_dir)
    out["id_path"] = props.get("ID_PATH", "")
    out["id_path_tag"] = props.get("ID_PATH_TAG", "")
    return out


def camera_instance_key(ident: dict[str, str]) -> str:
    if ident.get("id_path_tag"):
        return ident["id_path_tag"]
    if ident.get("kernel_usb_devpath"):
        return ident["kernel_usb_devpath"]
    if ident.get("vendor") and ident.get("product"):
        base = f"{ident['vendor']}:{ident['product']}"
        if ident.get("serial"):
            return f"{base}:{ident['serial']}"
        return base
    return ident.get("phys", "") or "(unknown)"


def format_camera_identity_line(ident: dict[str, str]) -> str:
    bits: list[str] = []
    if ident.get("id_path_tag"):
        bits.append(f"instance={ident['id_path_tag']}")
    elif ident.get("kernel_usb_devpath"):
        bits.append(f"sysfs={ident['kernel_usb_devpath']}")
    else:
        fk = camera_instance_key(ident)
        if fk and fk != "(unknown)":
            bits.append(f"key={fk}")

    if ident.get("vendor") and ident.get("product"):
        core = f"{ident['vendor']}:{ident['product']}"
        if ident.get("serial"):
            core = f"{core} · usb_serial={ident['serial']}"
        bits.append(core)
        if ident.get("usb_port_id"):
            bits.append(f"@{ident['usb_port_id']}")
        if ident.get("manufacturer") or ident.get("product_name"):
            bits.append(f"| {ident.get('manufacturer', '')} {ident.get('product_name', '')}".strip())
        return " ".join(bits)
    if ident.get("phys"):
        return f"non-usb {ident['phys']}"
    return "(unknown)"


def _norm_hex(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    if t.startswith("0x"):
        t = t[2:]
    return t


def identity_matches_select(ident: dict[str, str], spec: dict[str, str] | str) -> bool:
    line = format_camera_identity_line(ident).lower()
    if isinstance(spec, str):
        return spec.lower() in line
    if not isinstance(spec, dict):
        return False
    for key, want in spec.items():
        if want is None or want == "":
            continue
        w = str(want).strip()
        if key in ("vendor", "product"):
            if _norm_hex(w) != _norm_hex(ident.get(key, "")):
                return False
        elif key == "serial":
            if ident.get("serial", "") != w:
                return False
        elif key == "usb_port_id":
            if ident.get("usb_port_id", "") != w:
                return False
        elif key == "manufacturer":
            if w.lower() not in (ident.get("manufacturer") or "").lower():
                return False
        elif key == "product_name":
            if w.lower() not in (ident.get("product_name") or "").lower():
                return False
        elif key == "id_path":
            got = ident.get("id_path", "")
            if got != w and w not in got:
                return False
        elif key in ("id_path_tag", "instance"):
            if ident.get("id_path_tag", "") != w:
                return False
        elif key == "kernel_usb_devpath":
            got = ident.get("kernel_usb_devpath", "")
            if got != w and w not in got:
                return False
        elif key == "instance_key":
            if camera_instance_key(ident) != w:
                return False
        else:
            return False
    return True


def open_capture(dev: int | str):
    import cv2

    if isinstance(dev, int):
        return cv2.VideoCapture(dev)

    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(dev, getattr(cv2, "CAP_V4L2", 200))
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(dev)


def read_valid_frame(cap) -> bool:
    ret, img = cap.read()
    if not ret or img is None or getattr(img, "size", 0) == 0:
        return False
    h, w = img.shape[:2]
    return w > 2 and h > 2


def probe_cameras_detailed(*, warmup_reads: int = WARMUP_READS) -> list[dict[str, Any]]:
    """One entry per physical camera that delivers frames."""
    found: list[dict[str, Any]] = []
    seen_physical: set[str | tuple[str, int]] = set()

    if sys.platform.startswith("linux"):
        candidates: list[str | int] = list_v4l2_device_nodes()
    else:
        candidates = list(range(8))

    for dev_path in candidates:
        if isinstance(dev_path, str) and dev_path.startswith("/dev/") and not os.path.exists(dev_path):
            continue

        cap = open_capture(dev_path)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(max(1, warmup_reads)):
            if read_valid_frame(cap):
                ok = True
                break
        if not ok:
            cap.release()
            continue

        if isinstance(dev_path, str) and "/dev/video" in dev_path:
            phys = get_v4l2_physical_device_key(dev_path)
            key: str | tuple[str, int] = phys if phys is not None else dev_path
        else:
            key = ("index", int(dev_path))
        if key in seen_physical:
            cap.release()
            continue
        seen_physical.add(key)

        ident = get_camera_identity(dev_path) if isinstance(dev_path, str) else {}
        found.append({"path": dev_path, "identity": ident})
        cap.release()

    return found


def resolve_cameras_from_select(
    detailed: list[dict[str, Any]],
    select_list: list[dict[str, str] | str],
) -> list[dict[str, Any]]:
    remaining = list(detailed)
    chosen: list[dict[str, Any]] = []
    for spec in select_list:
        hit = None
        for i, entry in enumerate(remaining):
            if identity_matches_select(entry.get("identity") or {}, spec):
                hit = remaining.pop(i)
                break
        if hit is None:
            wanted = repr(spec)
            avail = [
                f"{entry['path']} -> {format_camera_identity_line(entry.get('identity') or {})}"
                for entry in detailed
            ]
            msg = (
                f"No camera matched {wanted}.\n"
                f"Working cameras:\n  " + ("\n  ".join(avail) if avail else "(none)")
            )
            raise ValueError(msg)
        chosen.append(hit)
    return chosen


def list_sysfs_camera_lines() -> list[str]:
    lines: list[str] = []
    if not sys.platform.startswith("linux"):
        return lines
    for dev_path in list_v4l2_device_nodes():
        if not os.path.exists(dev_path):
            continue
        ident = get_camera_identity(dev_path)
        lines.append(f"{dev_path} -> {format_camera_identity_line(ident)}")
    return lines


def print_list_cameras_help() -> None:
    lines = list_sysfs_camera_lines()
    if not lines:
        print("No /dev/video* nodes found (or not Linux).")
        return
    for line in lines:
        print(line)
    print(
        "\nEdit record/camera_ports.json with id_path_tag per role "
        "(cam_head, cam_left_wrist, cam_right_wrist).\n"
        "Run with --list-cameras-working to probe which nodes actually capture frames."
    )


def print_working_cameras() -> None:
    detailed = probe_cameras_detailed()
    if not detailed:
        print("No working V4L2 capture devices found.")
        return
    for entry in detailed:
        print(f"{entry['path']} -> {format_camera_identity_line(entry.get('identity') or {})}")


def configure_capture(cap, width: int, height: int, *, prefer_mjpeg: bool = True) -> None:
    import cv2

    if prefer_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def device_to_path(dev: int | str) -> str:
    if isinstance(dev, str):
        return dev if dev.startswith("/dev/") else dev
    return f"/dev/video{dev}"


def parse_camera_device_arg(value: str) -> int | str:
    """Parse CLI value: integer index or /dev/video path."""
    v = value.strip()
    if v.isdigit():
        return int(v)
    return v


def resolve_role_camera_devices(
    *,
    roles: tuple[str, ...] = THREE_CAM_ROLES,
    use_usb_ports: bool = True,
    ports_config: Path | None = None,
    overrides: dict[str, int | str | None] | None = None,
) -> dict[str, int | str]:
    """
    Return role -> device (path or index) for opening cameras.

    When ``use_usb_ports`` is True on Linux, roles without explicit overrides are
    resolved via ``camera_ports.json``. Overrides win per role.
    """
    overrides = overrides or {}
    out: dict[str, int | str] = {}

    need_usb: list[str] = []
    for role in roles:
        ov = overrides.get(role)
        if ov is not None:
            out[role] = ov
        elif use_usb_ports and sys.platform.startswith("linux"):
            need_usb.append(role)
        else:
            raise ValueError(
                f"No device for {role!r}: set --video-cam-* or enable --use-usb-camera-ports "
                f"(Linux + record/camera_ports.json)."
            )

    if not need_usb:
        return out

    role_select = load_role_usb_select(ports_config)
    missing_roles = [r for r in need_usb if r not in role_select]
    if missing_roles:
        raise ValueError(
            f"camera_ports.json missing roles: {missing_roles}. "
            f"Edit {ports_config or _DEFAULT_PORTS_FILE} after --list-cameras."
        )

    select_list = [role_select[r] for r in need_usb]
    detailed = probe_cameras_detailed()
    if not detailed:
        raise RuntimeError(
            "No working V4L2 cameras found. Check USB cables; run --list-cameras-working."
        )

    matched = resolve_cameras_from_select(detailed, select_list)
    for role, entry in zip(need_usb, matched, strict=True):
        out[role] = entry["path"]

    print("[usb_cameras] Resolved by USB port:", flush=True)
    for role in roles:
        dev = out[role]
        ident = get_camera_identity(dev) if isinstance(dev, str) else {}
        tag = ident.get("id_path_tag", "")
        print(f"  {role}: {device_to_path(dev)}  instance={tag or '?'}", flush=True)

    return out


def add_three_camera_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-usb-camera-ports",
        action=argparse.BooleanOptionalAction,
        default=sys.platform.startswith("linux"),
        help="Pin cameras by USB port (record/camera_ports.json). Default: on for Linux.",
    )
    parser.add_argument(
        "--camera-ports-config",
        type=Path,
        default=_DEFAULT_PORTS_FILE,
        help="JSON mapping cam_head / cam_left_wrist / cam_right_wrist to USB match specs.",
    )
    parser.add_argument(
        "--video-cam-head",
        type=parse_camera_device_arg,
        default=None,
        help="Override head camera (int index or /dev/videoN). Skips USB port pin for this role.",
    )
    parser.add_argument(
        "--video-cam-left-wrist",
        type=parse_camera_device_arg,
        default=None,
        help="Override left wrist camera.",
    )
    parser.add_argument(
        "--video-cam-right-wrist",
        type=parse_camera_device_arg,
        default=None,
        help="Override right wrist camera.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print sysfs USB identity per /dev/video* and exit.",
    )
    parser.add_argument(
        "--list-cameras-working",
        action="store_true",
        help="Open each /dev/video* and list devices that deliver frames.",
    )
    parser.add_argument(
        "--preview-cameras",
        action="store_true",
        help="Live 3-panel view labeled HEAD / LEFT WRIST / RIGHT WRIST (verify camera_ports.json). Esc=quit.",
    )
    parser.add_argument(
        "--preview-all-cameras",
        action="store_true",
        help="Cycle through every working camera with its USB port id (for building camera_ports.json).",
    )


def camera_cli_from_args(args: argparse.Namespace) -> dict[str, int | str]:
    return resolve_role_camera_devices(
        use_usb_ports=bool(args.use_usb_camera_ports),
        ports_config=args.camera_ports_config,
        overrides={
            "cam_head": args.video_cam_head,
            "cam_left_wrist": args.video_cam_left_wrist,
            "cam_right_wrist": args.video_cam_right_wrist,
        },
    )


def _annotate_preview_frame(
    bgr: np.ndarray,
    *,
    title: str,
    subtitle: str,
    banner_bgr: tuple[int, int, int],
) -> np.ndarray:
    import cv2

    out = bgr.copy()
    h, w = out.shape[:2]
    bar_h = max(36, h // 12)
    cv2.rectangle(out, (0, 0), (w, bar_h), banner_bgr, thickness=-1)
    cv2.putText(
        out,
        title,
        (8, bar_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            out,
            subtitle,
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return out


def run_camera_mapping_preview(
    device_map: dict[str, int | str],
    *,
    image_shape: tuple[int, int] = (480, 640),
    window_name: str = "GR00T: verify camera_ports.json (Esc=quit)",
) -> None:
    """
    Show HEAD | LEFT WRIST | RIGHT WRIST side-by-side using the resolved device map.

    Move each physical camera / wave your hand to confirm the label matches the view.
    """
    import cv2

    banner_colors = {
        "cam_head": (40, 160, 40),
        "cam_left_wrist": (180, 100, 30),
        "cam_right_wrist": (30, 90, 200),
    }
    rig = USBCameraRig(device_map, image_shape, prefer_mjpeg=False)
    print("Preview open. Check each panel matches HEAD / LEFT WRIST / RIGHT WRIST.", flush=True)
    print("Press Esc in the preview window to quit.", flush=True)
    try:
        while True:
            panels: list[np.ndarray] = []
            for role in THREE_CAM_ROLES:
                rgb = rig.read_rgb(role)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                dev = device_map[role]
                path = device_to_path(dev)
                ident = get_camera_identity(path) if isinstance(path, str) else {}
                tag = ident.get("id_path_tag", "?")
                subtitle = f"{path}  instance={tag}"
                panels.append(
                    _annotate_preview_frame(
                        bgr,
                        title=ROLE_DISPLAY_LABELS.get(role, role),
                        subtitle=subtitle,
                        banner_bgr=banner_colors.get(role, (80, 80, 80)),
                    )
                )
            strip = np.hstack(panels)
            cv2.imshow(window_name, strip)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        rig.close()
        cv2.destroyAllWindows()


def run_preview_all_working_cameras(*, image_shape: tuple[int, int] = (480, 640)) -> None:
    """Show each working camera one at a time with USB instance id (build camera_ports.json)."""
    import cv2

    detailed = probe_cameras_detailed()
    if not detailed:
        print("No working cameras found.")
        return

    print("Press Esc to quit, Space or n for next camera.", flush=True)
    idx = 0
    h, w = image_shape
    try:
        while True:
            entry = detailed[idx % len(detailed)]
            path = entry["path"]
            ident = entry.get("identity") or {}
            tag = ident.get("id_path_tag", "?")
            title = f"Camera {idx % len(detailed) + 1}/{len(detailed)}"
            subtitle = f"{path}  instance={tag}"

            cap = open_capture(path)
            if not cap.isOpened():
                print(f"Skip {path} (not opened)")
                idx += 1
                continue
            configure_capture(cap, w, h, prefer_mjpeg=False)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        blank = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        if frame.shape[:2] != (h, w):
                            blank = cv2.resize(frame, (w, h))
                        else:
                            blank = frame
                    view = _annotate_preview_frame(
                        blank,
                        title=title,
                        subtitle=subtitle,
                        banner_bgr=(60, 60, 140),
                    )
                    cv2.imshow("GR00T: discover USB camera ports (Esc=quit, Space=next)", view)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        return
                    if key in (32, ord("n"), ord("N")):
                        break
            finally:
                cap.release()
            idx += 1
    finally:
        cv2.destroyAllWindows()


def run_camera_preview_from_args(args: argparse.Namespace) -> None:
    """Entry for --preview-cameras / --preview-all-cameras."""
    h = int(getattr(args, "image_height", 480))
    w = int(getattr(args, "image_width", 640))
    shape = (h, w)
    if getattr(args, "preview_all_cameras", False):
        run_preview_all_working_cameras(image_shape=shape)
        return
    try:
        device_map = camera_cli_from_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        print(
            "\nA role in camera_ports.json does not match any plugged-in camera.\n"
            "Run:  uv run python record/preview_three_cameras.py --preview-all-cameras\n"
            "Then update record/camera_ports.json with the correct instance= id_path_tag.",
            flush=True,
        )
        raise SystemExit(2) from exc
    run_camera_mapping_preview(device_map, image_shape=shape)


def handle_camera_list_flags(args: argparse.Namespace) -> bool:
    """If list/preview flags set, run and return True (caller should exit)."""
    if getattr(args, "list_cameras", False):
        print_list_cameras_help()
        return True
    if getattr(args, "list_cameras_working", False):
        print_working_cameras()
        return True
    if getattr(args, "preview_cameras", False) or getattr(args, "preview_all_cameras", False):
        run_camera_preview_from_args(args)
        return True
    return False


class USBCameraRig:
    """Open and read a fixed set of named USB cameras."""

    def __init__(
        self,
        device_map: dict[str, int | str],
        image_shape: tuple[int, int],
        *,
        prefer_mjpeg: bool | None = None,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for USB cameras") from exc

        self._cv2 = cv2
        self.image_shape = image_shape
        self._caps: dict[str, Any] = {}
        multi = len(device_map) > 1
        if prefer_mjpeg is None:
            prefer_mjpeg = not multi

        h, w = image_shape
        for cam_name, dev in device_map.items():
            path = device_to_path(dev) if isinstance(dev, int) else str(dev)
            cap = open_capture(dev)
            if not cap.isOpened():
                cap.release()
                cap = open_capture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {cam_name} at {path}")
            configure_capture(cap, w, h, prefer_mjpeg=prefer_mjpeg)
            self._caps[cam_name] = cap
            time.sleep(0.2)

    def read_rgb(self, cam_name: str) -> np.ndarray:
        cv2 = self._cv2
        h, w = self.image_shape
        cap = self._caps[cam_name]
        ret, frame = cap.read()
        if not ret or frame is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

    def read_all_rgb(self, camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {cam: self.read_rgb(cam) for cam in camera_names}

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
