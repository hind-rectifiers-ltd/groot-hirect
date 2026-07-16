import argparse
import glob
import os
import pickle
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import cv2
import numpy as np

# Stream to multi_img_server.py on the viewing PC (UDP + per-camera label).
SERVER_IP = "192.168.1.92"
SERVER_PORT = 6668
STREAM_UDP = True
JPEG_QUALITY = 25
SOCKET_BUF_SIZE = 2 * 1024 * 1024

# Each camera runs in its own thread; the main thread only calls imshow/waitKey.
WARMUP_READS = 15
THREAD_START_DELAY_S = 0.25

# Linux: pin cameras by USB identity instead of /dev/videoN (numbers move across reboots).
# Leave empty to auto-detect every working camera (original behavior).
# Each entry is matched in order; use: python multi_img_client.py --list-cameras
# Examples:
#   CAMERA_SELECT = ["0c45:0261"]                      # substring match on the identity line
#   CAMERA_SELECT = [{"vendor": "0c45", "product": "0261"}]
# Prefer udev instance IDs (unique on this PC even when USB iSerial is duplicated):
#   CAMERA_SELECT = [
#       {"id_path_tag": "pci-0000_00_14_0-usb-0_2"},
#       {"id_path_tag": "pci-0000_00_14_0-usb-0_3"},
#       {"id_path_tag": "pci-0000_00_14_0-usb-0_8"},
#   ]
CAMERA_SELECT = []

# Optional friendly names for OpenCV preview windows (and UDP packet labels).
# Keys are id_path_tag from `python multi_img_client.py --list-cameras` (stable per USB port).
# Titles show as "Left (video0)" — role plus current V4L device node.
CAMERA_PREVIEW_NAMES = {
    "pci-0000_00_14_0-usb-0_2": "Left",
    "pci-0000_00_14_0-usb-0_3": "Right",
    "pci-0000_00_14_0-usb-0_8": "Head",
}

# Cache udev properties per USB sysfs directory (many /dev/video* nodes share one USB device).
_UDEV_USB_CACHE = {}


def get_screen_size():
    """
    Return primary monitor size as (width, height), or None if unavailable.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        if screen_w > 0 and screen_h > 0:
            return screen_w, screen_h
    except Exception:
        pass
    return None


def pick_frame_size(num_cameras):
    """Smaller frames when multiple UVC streams share USB bandwidth."""
    if num_cameras <= 1:
        return 640, 480
    return 640, 640


def list_v4l2_device_nodes():
    """Return sorted /dev/video* paths that exist (Linux V4L2)."""
    paths = glob.glob("/dev/video[0-9]*")
    def sort_key(p):
        m = re.search(r"(\d+)$", p)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=sort_key)


def get_v4l2_physical_device_key(dev_path):
    """Map /dev/videoN -> sysfs realpath of V4L2 device node (dedupe metadata nodes)."""
    m = re.search(r"video(\d+)$", dev_path)
    if not m:
        return None
    n = m.group(1)
    link = "/sys/class/video4linux/video{}/device".format(n)
    try:
        if os.path.exists(link):
            return os.path.realpath(link)
    except OSError:
        pass
    return None


def _read_sysfs_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def find_usb_device_sysfs_dir(device_realpath):
    """
    Walk up from a V4L2 device path until we find a directory with idVendor/idProduct
    (USB device, not interface).
    """
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


def _udev_usb_properties(usb_dir):
    """
    Return udev properties for a USB device sysfs path (cached).
    Used for ID_PATH / ID_PATH_TAG — stable unique IDs on this host for each USB attachment,
    even when the USB iSerial string is missing or duplicated across units.
    """
    rp = os.path.realpath(usb_dir)
    if rp in _UDEV_USB_CACHE:
        return _UDEV_USB_CACHE[rp]
    props = {}
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


def get_camera_identity(dev_path):
    """
    Return stable identifiers for a V4L2 device (Linux).

    USB webcams should expose a unique iSerial; many vendors reuse the same string on every
    unit, so it is not reliable for telling two cameras apart. Linux/udev assign:
      - ID_PATH / ID_PATH_TAG: unique for this physical USB port chain on this PC (preferred).
      - kernel_usb_devpath: sysfs path under /devices/... (unique fallback without udev).
    """
    phys = get_v4l2_physical_device_key(dev_path)
    out = {
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
    if usb_dir:
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


def camera_instance_key(ident):
    """Best stable unique key for this USB device on this machine (for ordering / selection)."""
    if ident.get("id_path_tag"):
        return ident["id_path_tag"]
    if ident.get("kernel_usb_devpath"):
        return ident["kernel_usb_devpath"]
    if ident.get("vendor") and ident.get("product"):
        base = "{}:{}".format(ident["vendor"], ident["product"])
        if ident.get("serial"):
            return "{}:{}".format(base, ident["serial"])
        return base
    return ident.get("phys", "") or "(unknown)"


def format_camera_identity_line(ident):
    """Single-line summary for printing and substring matching."""
    bits = []
    if ident.get("id_path_tag"):
        bits.append("instance={}".format(ident["id_path_tag"]))
    elif ident.get("kernel_usb_devpath"):
        bits.append("sysfs={}".format(ident["kernel_usb_devpath"]))
    else:
        fk = camera_instance_key(ident)
        if fk and fk != "(unknown)":
            bits.append("key={}".format(fk))

    if ident.get("vendor") and ident.get("product"):
        core = "{}:{}".format(ident["vendor"], ident["product"])
        if ident.get("serial"):
            core = "{} · usb_serial={}".format(core, ident["serial"])
        bits.append(core)
        if ident.get("usb_port_id"):
            bits.append("@{}".format(ident["usb_port_id"]))
        if ident.get("manufacturer") or ident.get("product_name"):
            bits.append("| {} {}".format(ident.get("manufacturer", ""), ident.get("product_name", "")).strip())
        return " ".join(bits)
    if ident.get("phys"):
        return "non-usb {}".format(ident["phys"])
    return "(unknown)"


def _norm_hex(s):
    if s is None:
        return ""
    t = str(s).strip().lower()
    if t.startswith("0x"):
        t = t[2:]
    return t


def identity_matches_select(ident, spec):
    """
    spec: str (substring match on format_camera_identity_line, case-insensitive)
          or dict with any of: id_path_tag (recommended), id_path, kernel_usb_devpath,
          vendor, product, serial, manufacturer, product_name, usb_port_id.
    """
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
            got = _norm_hex(ident.get(key, ""))
            if _norm_hex(w) != got:
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


def open_capture(dev):
    """
    Open by /dev/video* path (Linux) or numeric index (other OS).
    Path + CAP_V4L2 avoids 'can't be used to capture by index' on some builds.
    """
    if isinstance(dev, int):
        return cv2.VideoCapture(dev)

    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(dev)


def read_valid_frame(cap):
    ret, img = cap.read()
    if not ret or img is None or getattr(img, "size", 0) == 0:
        return False
    h, w = img.shape[:2]
    return w > 2 and h > 2


def probe_cameras_detailed():
    """
    Return one entry per physical camera that delivers frames:
    [{"path": "/dev/videoN", "identity": {...}}, ...]
    Deduplicate by sysfs device path (skip metadata-only nodes sharing the same camera).
    """
    found = []
    seen_physical = set()

    if sys.platform.startswith("linux"):
        candidates = list_v4l2_device_nodes()
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
        for _ in range(WARMUP_READS):
            if read_valid_frame(cap):
                ok = True
                break
        if not ok:
            cap.release()
            continue

        if isinstance(dev_path, str) and "/dev/video" in dev_path:
            phys = get_v4l2_physical_device_key(dev_path)
            key = phys if phys is not None else dev_path
        else:
            key = ("index", dev_path)
        if key in seen_physical:
            cap.release()
            continue
        seen_physical.add(key)

        ident = (
            get_camera_identity(dev_path)
            if (isinstance(dev_path, str) and sys.platform.startswith("linux"))
            else {}
        )
        found.append({"path": dev_path, "identity": ident})
        cap.release()

    return found


def probe_devices():
    """Backward-compatible: list of device paths only."""
    return [x["path"] for x in probe_cameras_detailed()]


def resolve_cameras_from_select(detailed, select_list):
    """
    Pick an ordered subset of probe_cameras_detailed() entries matching select_list.
    Each physical camera is used at most once.
    """
    remaining = list(detailed)
    chosen = []
    for spec in select_list:
        hit = None
        for i, entry in enumerate(remaining):
            if identity_matches_select(entry.get("identity") or {}, spec):
                hit = remaining.pop(i)
                break
        if hit is None:
            wanted = repr(spec)
            avail = [
                "{} -> {}".format(
                    entry["path"],
                    format_camera_identity_line(entry.get("identity") or {}),
                )
                for entry in detailed
            ]
            msg = "CAMERA_SELECT: no camera matched {}.\nWorking cameras:\n  {}".format(
                wanted,
                "\n  ".join(avail) if avail else "(none)",
            )
            raise ValueError(msg)
        chosen.append(hit)
    return chosen


def list_sysfs_camera_lines():
    """Fast listing without opening devices (Linux): sysfs identity per /dev/video* node."""
    lines = []
    if not sys.platform.startswith("linux"):
        return lines
    for dev_path in list_v4l2_device_nodes():
        if not os.path.exists(dev_path):
            continue
        ident = get_camera_identity(dev_path)
        lines.append(
            "{} -> {}".format(dev_path, format_camera_identity_line(ident))
        )
    return lines


def configure_capture(cap, width, height, prefer_mjpeg=True):
    """
    For multiple USB cameras, forcing MJPEG on all streams often starves the bus;
    the second device may open but never produce frames. Prefer uncompressed or
    driver default when prefer_mjpeg is False.
    """
    if prefer_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def display_label_for_entry(entry):
    """Short window / UDP label including USB identity when known."""
    p = entry["path"]
    ident = entry.get("identity") or {}
    base = os.path.basename(p) if isinstance(p, str) else "camera{}".format(p)
    line = format_camera_identity_line(ident)
    if not line or line == "(unknown)":
        return base
    full = "{} | {}".format(base, line)
    if len(full) > 120:
        full = full[:117] + "..."
    return full


def preview_label_for_entry(entry):
    """User-facing preview label: CAMERA_PREVIEW_NAMES[id_path_tag] plus /dev node (e.g. video0)."""
    p = entry.get("path")
    dev_short = os.path.basename(p) if isinstance(p, str) else "camera{}".format(p)
    ident = entry.get("identity") or {}
    tag = ident.get("id_path_tag", "")
    if tag and CAMERA_PREVIEW_NAMES and tag in CAMERA_PREVIEW_NAMES:
        return "{} ({})".format(CAMERA_PREVIEW_NAMES[tag], dev_short)
    return display_label_for_entry(entry)


def unique_labels_for_paths(camera_entries):
    """Ensure OpenCV window names are unique when two cameras share the same label."""
    path_to_label = {}
    seen_count = {}
    for e in camera_entries:
        p = e["path"]
        base = preview_label_for_entry(e)
        n = seen_count.get(base, 0)
        seen_count[base] = n + 1
        if n == 0:
            path_to_label[p] = base
        else:
            path_to_label[p] = "{} #{}".format(base, n + 1)
    return path_to_label


def capture_loop(
    dev_path,
    stop_event,
    frames,
    lock,
    width,
    height,
    prefer_mjpeg=True,
    sock=None,
    server_addr=None,
    stream_udp=False,
    display_label=None,
):
    label = display_label or window_title(dev_path)
    cap = open_capture(dev_path)
    if not cap.isOpened():
        print("Thread {}: could not open {}".format(label, dev_path))
        return
    configure_capture(cap, width, height, prefer_mjpeg=prefer_mjpeg)

    while not stop_event.is_set():
        ret, img = cap.read()
        if not ret or img is None or getattr(img, "size", 0) == 0:
            continue
        with lock:
            frames[dev_path] = img.copy()

        if stream_udp and sock is not None and server_addr is not None:
            ok, buffer = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if ok:
                tag = label
                b = tag.encode("utf-8")
                if len(b) > 255:
                    b = b[:255]
                header = struct.pack("!B", len(b)) + b
                payload = pickle.dumps(buffer)
                sock.sendto(header + payload, server_addr)

    cap.release()


def window_title(dev_path):
    return os.path.basename(dev_path) if isinstance(dev_path, str) else "camera{}".format(dev_path)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera capture / UDP stream (multi_img_server).")
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print sysfs USB identity for each /dev/video* node and exit (no capture).",
    )
    return p.parse_args()


def placeholder_frame(width, height, label):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        label,
        (8, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img


def main():
    args = parse_args()
    if args.list_cameras:
        lines = list_sysfs_camera_lines()
        if not lines:
            print("No /dev/video* nodes found (or not Linux).")
        else:
            for line in lines:
                print(line)
            print(
                "\nPrefer matching instance=… (udev ID_PATH_TAG) — unique per USB slot on this PC. "
                "USB usb_serial=… is often duplicated on identical webcams.\n"
                "CAMERA_SELECT dict keys: id_path_tag (or instance), id_path, kernel_usb_devpath, "
                "vendor, product, serial, usb_port_id, manufacturer, product_name."
            )
        return

    detailed = probe_cameras_detailed()
    if CAMERA_SELECT:
        try:
            camera_entries = resolve_cameras_from_select(detailed, CAMERA_SELECT)
        except ValueError as e:
            print(str(e))
            raise SystemExit(2)
    else:
        camera_entries = detailed

    devices = [e["path"] for e in camera_entries]
    if not devices:
        print("Error: No working V4L2 capture devices found under /dev/video*.")
        raise SystemExit(1)

    path_label = unique_labels_for_paths(camera_entries)

    width, height = pick_frame_size(len(devices))
    print("Using {}x{} for {} camera(s):".format(width, height, len(devices)))
    for e in camera_entries:
        pth = e["path"]
        print("  ", pth, "->", format_camera_identity_line(e.get("identity") or {}))
    print("Press Esc in any window to quit.")

    sock = None
    server_addr = (SERVER_IP, SERVER_PORT)
    if STREAM_UDP:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUF_SIZE)
        print("UDP stream -> {}:{} (multi_img_server.py)".format(SERVER_IP, SERVER_PORT))

    stop_event = threading.Event()
    frames = {}
    lock = threading.Lock()
    threads = []
    multi = len(devices) > 1

    for dev_path in devices:
        t = threading.Thread(
            target=capture_loop,
            args=(
                dev_path,
                stop_event,
                frames,
                lock,
                width,
                height,
                not multi,
                sock,
                server_addr,
                STREAM_UDP,
                path_label.get(dev_path),
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(THREAD_START_DELAY_S)

    last_shown = {}
    try:
        screen_size = get_screen_size()
        for i, dev_path in enumerate(devices):
            title = path_label.get(dev_path, window_title(dev_path))
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            ph = placeholder_frame(width, height, "waiting " + title)
            cv2.imshow(title, ph)
            if screen_size is not None:
                screen_w, screen_h = screen_size
                tile_w = max(320, screen_w // len(devices))
                tile_h = max(240, screen_h // 2)
                cv2.resizeWindow(title, tile_w, tile_h)
                cv2.moveWindow(title, i * tile_w, 0)
            else:
                cv2.moveWindow(title, i * (width + 40), 40)

        while True:
            with lock:
                snapshot = {k: v.copy() for k, v in frames.items()}
            for dev_path in devices:
                title = path_label.get(dev_path, window_title(dev_path))
                img = snapshot.get(dev_path)
                if img is not None:
                    last_shown[dev_path] = img
                    cv2.imshow(title, img)
                else:
                    cv2.imshow(
                        title,
                        last_shown.get(
                            dev_path,
                            placeholder_frame(width, height, "no signal " + title),
                        ),
                    )
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)
        cv2.destroyAllWindows()
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    main()
