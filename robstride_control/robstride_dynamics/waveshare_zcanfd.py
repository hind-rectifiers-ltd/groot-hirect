"""
python-can backend for Waveshare USB-CAN-FD-B (libcontrolcanfd.so).

RobStride motors use classic CAN 2.0 @ 1 Mbps (extended frames), not CAN FD.
Hardware label CAN1 = software channel 0, CAN2 = software channel 1.
"""

from __future__ import annotations

import os
import threading
import time
from ctypes import (
    CDLL,
    Structure,
    Union,
    byref,
    c_long,
    c_ubyte,
    c_uint,
    c_ulong,
    c_ulonglong,
    c_ushort,
    c_void_p,
    cdll,
)
from typing import Optional, Tuple

import can
from can import BusABC, Message

USBCANFD_200U = 41
STATUS_OK = 1
INVALID_DEVICE_HANDLE = 0
INVALID_CHANNEL_HANDLE = 0
TYPE_CAN = 0
TYPE_CANFD = 1

# __file__ is robstride_control/robstride_dynamics/waveshare_zcanfd.py
# repo root = ../../  (NOT ../../../ which escapes into projects/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_LIB_CANDIDATES = [
    os.environ.get("WAVESHARE_ZCANFD_LIB", ""),
    # Jetson / aarch64 SDK build (built from Jetson-nano/python3/libcontrolcanfd.a)
    os.path.join(
        _REPO_ROOT,
        "USB-CAN-FD-B-Linux",
        "Jetson-nano",
        "python3",
        "libcontrolcanfd.so",
    ),
    os.path.join(
        _REPO_ROOT,
        "USB-CAN-FD-B-Linux",
        "Raspberry",
        "python3",
        "libcontrolcanfd.so",
    ),
    os.path.join(
        _REPO_ROOT,
        "USB-CAN-FD-B-Linux",
        "VMware",
        "x86-python3",
        "libcontrolcanfd.so",
    ),
    os.path.join(
        _REPO_ROOT,
        "USB-CAN-FD-B-Linux",
        "VMware",
        "x86-c",
        "libcontrolcanfd.so",
    ),
    os.path.join(_REPO_ROOT, "scripts", "libcontrolcanfd.so"),
    "./libcontrolcanfd.so",
]


class _ZCAN_CHANNEL_CAN_INIT_CONFIG(Structure):
    _fields_ = [
        ("acc_code", c_uint),
        ("acc_mask", c_uint),
        ("reserved", c_uint),
        ("filter", c_ubyte),
        ("timing0", c_ubyte),
        ("timing1", c_ubyte),
        ("mode", c_ubyte),
    ]


class _ZCAN_CHANNEL_CANFD_INIT_CONFIG(Structure):
    _fields_ = [
        ("acc_code", c_uint),
        ("acc_mask", c_uint),
        ("abit_timing", c_uint),
        ("dbit_timing", c_uint),
        ("brp", c_uint),
        ("filter", c_ubyte),
        ("mode", c_ubyte),
        ("pad", c_ushort),
        ("reserved", c_uint),
    ]


class _ZCAN_CHANNEL_INIT_CONFIG(Union):
    _fields_ = [
        ("can", _ZCAN_CHANNEL_CAN_INIT_CONFIG),
        ("canfd", _ZCAN_CHANNEL_CANFD_INIT_CONFIG),
    ]


class ZCAN_CHANNEL_INIT_CONFIG(Structure):
    _fields_ = [("can_type", c_uint), ("config", _ZCAN_CHANNEL_INIT_CONFIG)]


class ZCAN_CAN_FRAME(Structure):
    _fields_ = [
        ("can_id", c_uint, 29),
        ("err", c_uint, 1),
        ("rtr", c_uint, 1),
        ("eff", c_uint, 1),
        ("can_dlc", c_ubyte),
        ("__pad", c_ubyte),
        ("__res0", c_ubyte),
        ("__res1", c_ubyte),
        ("data", c_ubyte * 8),
    ]


class ZCAN_Transmit_Data(Structure):
    _fields_ = [("frame", ZCAN_CAN_FRAME), ("transmit_type", c_uint)]


class ZCAN_Receive_Data(Structure):
    _fields_ = [("frame", ZCAN_CAN_FRAME), ("timestamp", c_ulonglong)]


def _load_library(lib_path: Optional[str] = None) -> CDLL:
    candidates = []
    if lib_path:
        candidates.append(lib_path)
    candidates.extend(p for p in _DEFAULT_LIB_CANDIDATES if p)

    errors = []
    for path in candidates:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            errors.append(f"{path}: not found")
            continue
        try:
            lib = cdll.LoadLibrary(path)
            _configure_prototypes(lib)
            return lib
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    raise FileNotFoundError(
        "Could not load libcontrolcanfd.so. Tried:\n  - "
        + "\n  - ".join(errors)
        + "\nSet WAVESHARE_ZCANFD_LIB to the full path of the .so file."
    )


def _configure_prototypes(lib: CDLL) -> None:
    lib.ZCAN_OpenDevice.restype = c_void_p
    lib.ZCAN_OpenDevice.argtypes = (c_uint, c_uint, c_uint)

    lib.ZCAN_CloseDevice.argtypes = (c_void_p,)
    lib.ZCAN_SetAbitBaud.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_SetDbitBaud.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_SetCANFDStandard.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_SetResistanceEnable.argtypes = (c_void_p, c_ulong, c_ulong)

    lib.ZCAN_InitCAN.restype = c_void_p
    lib.ZCAN_InitCAN.argtypes = (c_void_p, c_ulong, c_void_p)
    lib.ZCAN_StartCAN.argtypes = (c_void_p,)
    lib.ZCAN_ResetCAN.argtypes = (c_void_p,)
    lib.ZCAN_ClearBuffer.argtypes = (c_void_p,)

    lib.ZCAN_Transmit.argtypes = (c_void_p, c_void_p, c_ulong)
    lib.ZCAN_GetReceiveNum.argtypes = (c_void_p, c_ulong)
    lib.ZCAN_Receive.argtypes = (c_void_p, c_void_p, c_ulong, c_long)

    lib.ZCAN_ClearFilter.argtypes = (c_void_p,)
    lib.ZCAN_AckFilter.argtypes = (c_void_p,)


# ---------------------------------------------------------------------------
# Shared USB device (one Waveshare adapter = one ZCAN_OpenDevice, two CAN ports)
# ---------------------------------------------------------------------------

_lib: CDLL | None = None
_lib_lock = threading.Lock()

_device_pools: dict[int, "_DevicePool"] = {}
_device_pools_lock = threading.Lock()


class _DevicePool:
    """Reference-counted ZCAN device handle shared by zcan0 + zcan1 bus instances."""

    __slots__ = ("lib", "handle", "device_index", "refcount")

    def __init__(self, lib: CDLL, handle, device_index: int) -> None:
        self.lib = lib
        self.handle = handle
        self.device_index = int(device_index)
        self.refcount = 0


def _get_lib(lib_path: Optional[str] = None) -> CDLL:
    global _lib
    with _lib_lock:
        if _lib is None:
            _lib = _load_library(lib_path)
        return _lib


def _acquire_device(device_index: int, lib_path: Optional[str] = None) -> _DevicePool:
    lib = _get_lib(lib_path)
    with _device_pools_lock:
        pool = _device_pools.get(device_index)
        if pool is None:
            handle = lib.ZCAN_OpenDevice(USBCANFD_200U, device_index, 0)
            if not handle:
                raise can.CanInitializationError(
                    "ZCAN_OpenDevice failed. Is USB-CAN-FD-B plugged in? "
                    "Check: lsusb | grep -i can"
                )
            pool = _DevicePool(lib, handle, device_index)
            _device_pools[device_index] = pool
        pool.refcount += 1
        return pool


def _release_device(device_index: int) -> None:
    with _device_pools_lock:
        pool = _device_pools.get(device_index)
        if pool is None:
            return
        pool.refcount -= 1
        if pool.refcount <= 0:
            try:
                pool.lib.ZCAN_CloseDevice(pool.handle)
            except Exception:
                pass
            del _device_pools[device_index]


class WaveshareZCanFdBus(BusABC):
    """Classic CAN bus over Waveshare USB-CAN-FD-B."""

    def __init__(
        self,
        channel: int | str = 0,
        bitrate: int = 2_000_000,
        device_index: int = 0,
        enable_resistance: bool = True,
        lib_path: Optional[str] = None,
        **kwargs,
    ):
        # Accept "0", "1", "waveshare0", "zcan1", etc.
        if isinstance(channel, str):
            digits = "".join(ch for ch in channel if ch.isdigit())
            if digits == "":
                raise ValueError(
                    f"Invalid Waveshare channel '{channel}'. Use waveshare0 or waveshare1."
                )
            channel = int(digits)

        self.channel_id = int(channel)
        self.bitrate = int(bitrate)
        self.device_index = int(device_index)
        self.enable_resistance = bool(enable_resistance)
        self._lib_path = lib_path

        self._pool: _DevicePool | None = None
        self._device = None
        self._channel = None

        super().__init__(channel=self.channel_id, bitrate=self.bitrate, **kwargs)
        self._open()

    def _open(self) -> None:
        self._pool = _acquire_device(self.device_index, self._lib_path)
        self._lib = self._pool.lib
        self._device = self._pool.handle

        try:
            self._open_channel()
        except Exception:
            if self._channel is None:
                _release_device(self.device_index)
                self._pool = None
                self._device = None
            raise

    def _open_channel(self) -> None:
        # RobStride needs 1 Mbps classic CAN. Dbit is unused for classic frames
        # but the firmware still expects both to be set before init.
        if self._lib.ZCAN_SetAbitBaud(self._device, self.channel_id, self.bitrate) != STATUS_OK:
            raise can.CanInitializationError(f"Set Abit baud {self.bitrate} failed")
        if self._lib.ZCAN_SetDbitBaud(self._device, self.channel_id, self.bitrate) != STATUS_OK:
            raise can.CanInitializationError(f"Set Dbit baud {self.bitrate} failed")

        # ISO CAN FD mode (harmless for classic frames)
        self._lib.ZCAN_SetCANFDStandard(self._device, self.channel_id, 0)

        if self.enable_resistance:
            # Software 120Ω enable (also flip the physical switch on the adapter)
            self._lib.ZCAN_SetResistanceEnable(self._device, self.channel_id, 1)

        init = ZCAN_CHANNEL_INIT_CONFIG()
        init.can_type = TYPE_CANFD
        init.config.canfd.acc_code = 0
        init.config.canfd.acc_mask = 0xFFFFFFFF
        init.config.canfd.filter = 1
        init.config.canfd.mode = 0  # normal mode
        init.config.canfd.brp = 0

        self._channel = self._lib.ZCAN_InitCAN(self._device, self.channel_id, byref(init))
        if not self._channel:
            raise can.CanInitializationError(f"ZCAN_InitCAN failed for channel {self.channel_id}")

        # Accept all IDs (critical — demo filter would drop motor replies)
        self._lib.ZCAN_ClearFilter(self._channel)
        self._lib.ZCAN_AckFilter(self._channel)

        if self._lib.ZCAN_StartCAN(self._channel) != STATUS_OK:
            self._channel = None
            raise can.CanInitializationError("ZCAN_StartCAN failed")

        self._lib.ZCAN_ClearBuffer(self._channel)
        print(
            f"Waveshare USB-CAN-FD-B ready: "
            f"channel={self.channel_id} (hardware CAN{self.channel_id + 1}), "
            f"bitrate={self.bitrate}"
        )

    def send(self, msg: Message, timeout: Optional[float] = None) -> None:
        if self._channel is None:
            raise can.CanOperationError("Bus is not open")

        tx = ZCAN_Transmit_Data()
        tx.transmit_type = 0  # normal send
        tx.frame.can_id = msg.arbitration_id & 0x1FFFFFFF
        tx.frame.err = 0
        tx.frame.rtr = 1 if msg.is_remote_frame else 0
        tx.frame.eff = 1 if msg.is_extended_id else 0
        tx.frame.can_dlc = msg.dlc if msg.dlc is not None else len(msg.data)

        data = bytes(msg.data)
        for i in range(tx.frame.can_dlc):
            tx.frame.data[i] = data[i] if i < len(data) else 0

        sent = self._lib.ZCAN_Transmit(self._channel, byref(tx), 1)
        if sent != 1:
            raise can.CanOperationError(f"ZCAN_Transmit failed (returned {sent})")

    def _recv_internal(self, timeout: Optional[float]) -> Tuple[Optional[Message], bool]:
        if self._channel is None:
            raise can.CanOperationError("Bus is not open")

        if timeout is None:
            wait_ms = -1  # block inside library when we call Receive
        elif timeout <= 0:
            wait_ms = 0
        else:
            wait_ms = max(1, int(timeout * 1000))

        # Fast path: nothing pending and non-blocking
        if wait_ms == 0 and self._lib.ZCAN_GetReceiveNum(self._channel, TYPE_CAN) <= 0:
            return None, False

        deadline = None if timeout is None else (time.time() + timeout)
        while True:
            pending = self._lib.ZCAN_GetReceiveNum(self._channel, TYPE_CAN)
            if pending > 0:
                break
            if timeout is not None and time.time() >= deadline:
                return None, False
            # Small poll; library wait_time alone is unreliable across builds
            time.sleep(0.001)

        rx = ZCAN_Receive_Data()
        got = self._lib.ZCAN_Receive(self._channel, byref(rx), 1, wait_ms)
        if got <= 0:
            return None, False

        dlc = rx.frame.can_dlc
        data = bytes(rx.frame.data[i] for i in range(dlc))
        msg = Message(
            arbitration_id=rx.frame.can_id & 0x1FFFFFFF,
            is_extended_id=bool(rx.frame.eff),
            is_remote_frame=bool(rx.frame.rtr),
            is_error_frame=bool(rx.frame.err),
            dlc=dlc,
            data=data,
            timestamp=rx.timestamp / 1_000_000.0 if rx.timestamp else time.time(),
            channel=self.channel_id,
        )
        return msg, False

    def shutdown(self) -> None:
        if self._channel:
            try:
                self._lib.ZCAN_ResetCAN(self._channel)
            except Exception:
                pass
            self._channel = None
        if self._pool is not None:
            _release_device(self.device_index)
            self._pool = None
            self._device = None
        super().shutdown()
