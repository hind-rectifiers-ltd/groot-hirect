#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobStride Advanced Control Example
Demonstrates advanced motor control techniques including trajectory planning
"""

import sys
import os
import time
import math
import numpy as np
from typing import List, Tuple

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from src.position_control import PositionControllerMIT
except ImportError:
    print("❌ Cannot import motor control modules")
    sys.exit(1)

class TrajectoryPlanner:
    """Simple trajectory planner for smooth motion"""

    def __init__(self, max_velocity: float = 2.0, max_acceleration: float = 5.0):
        self.max_velocity = max_velocity  # rad/s
        self.max_acceleration = max_acceleration  # rad/s²

    def plan_trajectory(self, start_pos: float, end_pos: float, duration: float) -> List[Tuple[float, float]]:
        """
        Plan a smooth trajectory from start_pos to end_pos
        Returns list of (position, time) tuples
        """
        # Generate time points
        dt = 0.02  # 50Hz update rate
        num_points = int(duration / dt)
        times = np.linspace(0, duration, num_points)

        # Cubic polynomial trajectory
        # s(t) = at³ + bt² + ct + d
        # Constraints: s(0)=start, s(T)=end, s'(0)=0, s'(T)=0

        T = duration
        d = start_pos
        a = 2 * (end_pos - start_pos) / (T**3)
        b = -3 * (end_pos - start_pos) / (T**2)
        c = 0

        positions = []
        for t in times:
            pos = a * t**3 + b * t**2 + c * t + d
            positions.append((pos, t))

        return positions

class AdvancedController:
    """Advanced motor controller with trajectory planning"""

    def __init__(self, motor_id: int):
        self.motor_id = motor_id
        self.controller = PositionControllerMIT(motor_id)
        self.planner = TrajectoryPlanner()

    def connect(self) -> bool:
        """Connect to motor"""
        return self.controller.connect()

    def execute_trajectory(self, target_angle: float, duration: float = 3.0):
        """Execute smooth trajectory to target position"""
        print(f"🎯 Planning trajectory to {target_angle}° over {duration}s")

        # Get current position (assume starting at 0 for simplicity)
        start_pos = 0.0  # This should be read from motor in real implementation
        end_pos = math.radians(target_angle)

        # Plan trajectory
        trajectory = self.planner.plan_trajectory(start_pos, end_pos, duration)

        print(f"📈 Executing {len(trajectory)} points...")

        # Execute trajectory
        for pos, t in trajectory:
            angle_deg = math.degrees(pos)
            self.controller.set_angle(angle_deg)
            time.sleep(0.02)  # Match trajectory planning rate

    def execute_sine_wave(self, amplitude: float = 45.0, frequency: float = 0.5, duration: float = 10.0):
        """Execute sinusoidal motion"""
        print(f"🌊 Executing sine wave: ±{amplitude}° at {frequency}Hz for {duration}s")

        start_time = time.time()
        dt = 0.02  # 50Hz update rate

        while time.time() - start_time < duration:
            t = time.time() - start_time
            angle = amplitude * math.sin(2 * math.pi * frequency * t)
            self.controller.set_angle(angle)
            time.sleep(dt)

    def execute_square_pattern(self, size: float = 90.0, duration: float = 2.0):
        """Execute square movement pattern"""
        print(f"⬜ Executing square pattern: {size}° sides, {duration}s per side")

        corners = [size, size, -size, -size, size]  # Return to start

        for corner in corners:
            print(f"📍 Moving to {corner}°")
            self.controller.set_angle(corner)
            time.sleep(duration)

    def calibrate_motor(self):
        """Motor calibration routine"""
        print("🔧 Starting motor calibration...")

        # Test different Kp/Kd values
        test_positions = [30.0, -30.0, 0.0]
        kp_values = [10.0, 30.0, 50.0]
        kd_values = [0.2, 0.5, 1.0]

        print("📊 Testing control parameters...")

        for kp in kp_values:
            for kd in kd_values:
                print(f"Testing Kp={kp}, Kd={kd}")
                self.controller.set_kp(kp)
                self.controller.set_kd(kd)

                # Test response
                start_time = time.time()
                self.controller.set_angle(test_positions[0])
                time.sleep(2.0)  # Wait for response

        print("✅ Calibration completed!")

    def stop_and_exit(self):
        """Stop motor and cleanup"""
        self.controller.stop_and_exit()

def main():
    """Main function demonstrating advanced control techniques"""
    print("🚀 RobStride Advanced Control Example")
    print("=" * 60)

    # Get motor ID from command line
    motor_id = 11
    if len(sys.argv) > 1:
        try:
            motor_id = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid motor ID. Using default (11)")

    print(f"Using Motor ID: {motor_id}")

    # Initialize controller
    controller = AdvancedController(motor_id)

    try:
        # Connect to motor
        if not controller.connect():
            print("❌ Failed to connect to motor")
            return 1

        print("✅ Connected successfully!")

        # Demonstration menu
        print("\n📋 Select demonstration:")
        print("1. Smooth trajectory planning")
        print("2. Sinusoidal motion")
        print("3. Square pattern movement")
        print("4. Motor calibration")
        print("5. All demonstrations (sequentially)")

        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            # Trajectory planning demo
            controller.execute_trajectory(180.0, 4.0)
            time.sleep(1.0)
            controller.execute_trajectory(-180.0, 4.0)
            controller.execute_trajectory(0.0, 2.0)

        elif choice == '2':
            # Sine wave demo
            controller.execute_sine_wave(amplitude=90.0, frequency=0.3, duration=15.0)

        elif choice == '3':
            # Square pattern demo
            controller.execute_square_pattern(size=60.0, duration=1.5)

        elif choice == '4':
            # Calibration demo
            controller.calibrate_motor()

        elif choice == '5':
            # All demos
            print("\n🎭 Running all demonstrations...")

            print("\n1️⃣ Trajectory Planning")
            controller.execute_trajectory(120.0, 3.0)
            time.sleep(0.5)
            controller.execute_trajectory(0.0, 2.0)
            time.sleep(1.0)

            print("\n2️⃣ Sinusoidal Motion")
            controller.execute_sine_wave(amplitude=45.0, frequency=0.5, duration=8.0)
            time.sleep(1.0)

            print("\n3️⃣ Square Pattern")
            controller.execute_square_pattern(size=30.0, duration=1.0)

        else:
            print("❌ Invalid choice")
            return 1

        print("\n🎉 Advanced control demonstration completed!")
        return 0

    except KeyboardInterrupt:
        print("\n👋 Demonstration interrupted by user")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1
    finally:
        controller.stop_and_exit()

if __name__ == "__main__":
    sys.exit(main())