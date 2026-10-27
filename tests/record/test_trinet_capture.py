"""Unit tests for Trinet IMU JPEG stripping and square crop (no cameras required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RECORD = Path(__file__).resolve().parents[2] / "record"
if str(_RECORD) not in sys.path:
    sys.path.insert(0, str(_RECORD))

from trinet_capture import crop_square_resize_rgb, strip_imu_jpeg  # noqa: E402


def _jpeg_with_imu() -> bytes:
    soi = b"\xff\xd8"
    app = b"\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 10
    fake_dht = b"\xff\xc4\x00\x04\x00\x00"
    imu = b"TRINETIMUSEI" + b"\x11" * 32
    real_dht = b"\xff\xc4\x00\x04\x00\x00"
    sos = b"\xff\xda\x00\x08\x01\x00\x00\x00\x00\x00"
    eoi = b"\xff\xd9"
    return soi + app + fake_dht + imu + real_dht + sos + eoi


def test_strip_imu_jpeg_removes_marker() -> None:
    raw = _jpeg_with_imu()
    assert b"TRINETIMUSEI" in raw
    cleaned = strip_imu_jpeg(raw)
    assert cleaned.startswith(b"\xff\xd8")
    assert b"TRINETIMUSEI" not in cleaned
    assert b"\xff\xda" in cleaned


def test_strip_imu_jpeg_passthrough_without_marker() -> None:
    buf = b"\xff\xd8\xff\xda\x00\x02\xff\xd9"
    assert strip_imu_jpeg(buf) == buf


def test_crop_square_resize_rgb_1080_from_1920x1080() -> None:
    bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bgr[:, 420:1500, 1] = 200  # green in the center 1080-wide band
    rgb = crop_square_resize_rgb(bgr, 1080, 1080)
    assert rgb.shape == (1080, 1080, 3)
    assert rgb.dtype == np.uint8
    # RGB order: green is channel 1
    assert int(rgb[:, :, 1].mean()) > 150


def test_crop_square_resize_rgb_720() -> None:
    bgr = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    rgb = crop_square_resize_rgb(bgr, 720, 720)
    assert rgb.shape == (720, 720, 3)


def test_crop_resize_rgb_1980x720_from_1920x1080() -> None:
    bgr = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    rgb = crop_square_resize_rgb(bgr, 720, 1980)
    assert rgb.shape == (720, 1980, 3)
    assert rgb.dtype == np.uint8
