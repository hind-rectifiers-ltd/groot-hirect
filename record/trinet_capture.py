"""Trinet Mono UVC cameras as decoded OpenCV frames.

Copied into this repo (do not import ``trinet_usb``). Each camera streams
1920x1080 MJPEG at 30 fps with a ``TRINETIMUSEI`` motion block spliced into the
JPEG, which makes OpenCV's decoder reject the frame. ``decode_trinet_raw``
removes that block and returns BGR.

Capture nodes are found by USB serial every start, so ``/dev/videoN`` changes
across replugs do not matter.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np

TRINET_VID = "2207"
TRINET_PID = "0016"
TRINET_NATIVE_WIDTH = 1920
TRINET_NATIVE_HEIGHT = 1080
TRINET_NATIVE_FPS = 30.0

# Record / policy default: square crop of the 1080p stream.
PREFERRED_OUTPUT_SIZE = 1080
MIN_OUTPUT_SIZE = 720

_IMU_MARK = b"TRINETIMUSEI"
_DHT = b"\xff\xc4"
_JPEG_SOI = b"\xff\xd8"


@dataclass(frozen=True)
class TrinetInfo:
    node: str
    serial: str
    by_id: str


def strip_imu_jpeg(buf: bytes) -> bytes:
    """Return a standard JPEG, with the in-band IMU block removed."""
    if not buf.startswith(_JPEG_SOI):
        return buf
    key = buf.find(_IMU_MARK)
    if key < 0:
        return buf
    fake = buf.rfind(_DHT, 0, key)
    if fake < 0:
        return buf
    start = key
    while True:
        real = buf.find(_DHT, start)
        if real < 0:
            return buf
        candidate = buf[:fake] + buf[real:]
        if b"\xff\xda" in candidate[fake : fake + 8192]:
            return candidate
        start = real + 2


def jpeg_bytes_from_raw(raw: np.ndarray) -> bytes | None:
    """Extract JPEG bytes from a CONVERT_RGB=0 VideoCapture buffer."""
    if raw is None or getattr(raw, "size", 0) == 0:
        return None
    if raw.ndim == 3 and raw.shape[0] > 1 and raw.shape[-1] == 3:
        # Already a BGR image (decoder ran). Not JPEG bytes.
        return None
    buf = np.ascontiguousarray(raw).reshape(-1).tobytes()
    if not buf.startswith(_JPEG_SOI):
        return None
    return strip_imu_jpeg(buf)


def decode_trinet_raw(raw: np.ndarray) -> np.ndarray | None:
    """Decode one V4L grab into BGR, stripping the IMU JPEG block if present."""
    if raw is None or getattr(raw, "size", 0) == 0:
        return None
    if raw.ndim == 3 and raw.shape[0] > 1 and raw.shape[-1] == 3:
        return raw
    jpeg = jpeg_bytes_from_raw(raw)
    if jpeg is None:
        return None
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    return frame


def crop_square_resize_rgb(bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center-crop the largest square, resize to ``(height, width)``, return RGB."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = bgr[y0 : y0 + side, x0 : x0 + side]
    if square.shape[0] != height or square.shape[1] != width:
        interp = cv2.INTER_AREA if side > max(height, width) else cv2.INTER_LINEAR
        square = cv2.resize(square, (width, height), interpolation=interp)
    return cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _usb_identity(node: str) -> tuple[str, str, str] | None:
    path = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(node)}/device")
    for _ in range(8):
        vendor = os.path.join(path, "idVendor")
        product = os.path.join(path, "idProduct")
        if os.path.exists(vendor) and os.path.exists(product):
            with open(vendor, encoding="utf-8", errors="replace") as f:
                vid = f.read().strip().lower()
            with open(product, encoding="utf-8", errors="replace") as f:
                pid = f.read().strip().lower()
            serial_path = os.path.join(path, "serial")
            serial = ""
            if os.path.exists(serial_path):
                with open(serial_path, encoding="utf-8", errors="replace") as f:
                    serial = f.read().strip()
            return vid, pid, serial
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def is_trinet_vid_pid(vid: str, pid: str) -> bool:
    return vid.lower() == TRINET_VID and pid.lower() == TRINET_PID


def list_trinet_cameras() -> list[TrinetInfo]:
    """Capture nodes only, ordered by serial so indexes stay stable across replugs."""
    found: list[TrinetInfo] = []
    by_id = {
        os.path.realpath(path): path
        for path in glob.glob("/dev/v4l/by-id/usb-Panoculon_Labs_Trinet_UVC_*-video-index0")
    }
    nodes = sorted(
        glob.glob("/dev/video*"),
        key=lambda n: int("".join(ch for ch in n if ch.isdigit()) or "0"),
    )
    by_serial: dict[str, list[tuple[str, str]]] = {}
    for node in nodes:
        ident = _usb_identity(node)
        if ident is None:
            continue
        vid, pid, serial = ident
        if not is_trinet_vid_pid(vid, pid):
            continue
        real = os.path.realpath(node)
        link = by_id.get(real, "")
        by_serial.setdefault(serial, []).append((real, link))
        if link:
            found.append(TrinetInfo(node=real, serial=serial, by_id=link))
    if not found:
        # No by-id capture links (some Jetson images omit them). Use the lowest
        # /dev/videoN per serial — that is the picture node, not metadata.
        for serial, pairs in by_serial.items():
            pairs.sort(key=lambda item: int("".join(ch for ch in item[0] if ch.isdigit()) or "0"))
            node, link = pairs[0]
            found.append(TrinetInfo(node=node, serial=serial, by_id=link))
    found.sort(key=lambda cam: cam.serial)
    return found


def is_trinet_capture_node(node: str | int | None) -> bool:
    if node is None:
        return False
    path = f"/dev/video{node}" if isinstance(node, int) else str(node)
    if not path.startswith("/dev/"):
        return False
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    return any(cam.node == real for cam in list_trinet_cameras())


def configure_trinet_capture(cap, *, fps: float | None = TRINET_NATIVE_FPS) -> None:
    """Native 1920x1080 MJPEG @ 30 fps, raw JPEG bytes (no OpenCV RGB convert)."""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TRINET_NATIVE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TRINET_NATIVE_HEIGHT)
    if fps is not None and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    # One slot makes a slow reader skip every other frame and lock to 15 fps.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)


class TrinetCamera:
    """Single-camera helper for probe / one-off reads."""

    def __init__(self, info: TrinetInfo, *, fps: float | None = TRINET_NATIVE_FPS):
        self.info = info
        self.cap = cv2.VideoCapture(info.node, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {info.node} (serial {info.serial})")
        configure_trinet_capture(self.cap, fps=fps)

    @property
    def serial(self) -> str:
        return self.info.serial

    @property
    def node(self) -> str:
        return self.info.node

    def read_bgr(self) -> np.ndarray | None:
        ok, raw = self.cap.read()
        if not ok or raw is None:
            return None
        return decode_trinet_raw(raw)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> TrinetCamera:
        return self

    def __exit__(self, *_) -> None:
        self.close()
