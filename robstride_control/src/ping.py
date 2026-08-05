#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobStride CAN bus scan / ping tool.

Usage:
  # Old SocketCAN adapter:
  sudo ip link set can0 up type can bitrate 1000000
  python3 ping.py can0

  # Waveshare USB-CAN-FD-B (no ip link — library opens the device):
  #   Hardware label CAN1 -> waveshare0
  #   Hardware label CAN2 -> waveshare1
  python3 ping.py waveshare0

This script pings IDs 1..254 and reports responding motors.
"""

import sys
import os
import time

# --- Import SDK ---
try:
    sdk_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    from robstride_dynamics import RobstrideBus
except ImportError:
    try:
        print("Package 'robstride_dynamics' not found, trying local import...")
        from bus import RobstrideBus
    except ImportError as e:
        print(f"Cannot import RobstrideBus SDK: {e}")
        print("Run from RobStride_Control/python, or: pip install -e .")
        sys.exit(1)


def main():
    if len(sys.argv) > 1:
        channel = sys.argv[1]
    else:
        channel = "waveshare0"

    print("RobStride bus scan")
    print(f"Channel: {channel}")
    if str(channel).lower().startswith(("waveshare", "zcan")):
        hw = int("".join(c for c in channel if c.isdigit()) or "0") + 1
        print(f"  -> Waveshare USB-CAN-FD-B hardware port CAN{hw}")
        print("  -> Classic CAN @ 1 Mbps (not CAN FD)")
        print("  -> No 'ip link' needed for this adapter")
    else:
        print("  -> SocketCAN interface (needs: sudo ip link set ... up)")
    print("Scanning motor IDs 1 .. 254")
    print("...")
    time.sleep(1)

    found_motors = None
    try:
        found_motors = RobstrideBus.scan_channel(channel, start_id=1, end_id=20)
    except Exception as e:
        print(f"\nScan error: {e}")
        if "Operation not permitted" in str(e) or "Permission denied" in str(e):
            print("Permission error: try sudo, or add udev rules for the USB device.")
            print(f"  Example: sudo python3 {sys.argv[0]} {channel}")
        elif "No such device" in str(e):
            print(f"Device error: SocketCAN interface '{channel}' not found.")
        elif "OpenDevice" in str(e) or "libcontrolcanfd" in str(e):
            print("Waveshare adapter not opened. Check:")
            print("  1) USB cable plugged in")
            print("  2) lsusb shows 'Microchip' / CANFD device")
            print("  3) lib path: USB-CAN-FD-B-Linux/VMware/x86-python3/libcontrolcanfd.so")
        sys.exit(1)

    if not found_motors:
        print("\nNo motors responded.")
        print("Checklist:")
        print("  - Motor powered on")
        print("  - H/L not swapped")
        print("  - Termination ON (adapter switch + motor end if needed)")
        print("  - Correct port: CAN1 -> waveshare0, CAN2 -> waveshare1")
        print("  - Bitrate 1 Mbps (already set by this tool)")
    else:
        print("\nScan done. Motors found:")
        print("=" * 60)
        print(f"{'Motor ID':<10} | {'MCU UUID':<45}")
        print("-" * 60)

        for motor_id in sorted(found_motors.keys()):
            _id, uuid = found_motors[motor_id]
            uuid_hex = uuid.hex()
            print(f"{motor_id:<10} | {uuid_hex}")

        print("=" * 60)


if __name__ == "__main__":
    main()
