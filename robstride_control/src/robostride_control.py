#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobStride Actuator Motion Control Script

This script demonstrates sinusoidal motion control with real-time position feedback,
error display, and mode switching capabilities.

Usage: python3 move_actuator.py <motor_id1> [motor_id2] [--channel can0] [--frequency 1.0] [--amplitude 1.0] [--mode 0|2]
"""

import sys
import os
import time
import math
import argparse
import signal
from typing import Optional

# Try to import SDK
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from robstride_dynamics import RobstrideBus, Motor, ParameterType, CommunicationType
except ImportError:
    try:
        from bus import RobstrideBus, Motor
        from protocol import ParameterType, CommunicationType
    except ImportError as e:
        print(f"❌ Failed to import SDK: {e}")
        sys.exit(1)


class RateLimiter:
    """Simple rate limiter to maintain control loop frequency."""
    def __init__(self, frequency: float):
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.last_time = time.time()
    
    def sleep(self):
        """Sleep to maintain the specified frequency."""
        elapsed = time.time() - self.last_time
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()


class MoveActuator:
    # Motor ID to model mapping
    MOTOR_MODEL_MAP = {
        2: "rs-03",
        4: "rs-03",
        6: "rs-06",
        8: "rs-06",
        10: "rs-02"
    }
    
    def __init__(self, motor_ids: list, c: str = 'can0', 
                 frequency: float = 1.0, amplitude: float = 0.0, mode: int = 0):
        self.motor_ids = motor_ids
        self.motor_names = [f"motor_{motor_id}" for motor_id in motor_ids]
        self.channel = c
        self.motion_frequency = frequency  # Hz
        self.motion_amplitude = amplitude  # radians
        
        self.bus: Optional[RobstrideBus] = None
        self.running = True
        
        # MIT mode control parameters - per motor
        self.kp = {}  # Stiffness (Nm/rad) - per motor
        self.kd = {}  # Damping (Nm/rad/s) - per motor
        self.torque_limit = {} # Nm
        self.tff= 0.0
        
        # Initialize per-motor kp and kd
        for motor_id, motor_name in zip(self.motor_ids, self.motor_names):
            if motor_id == 2 or motor_id == 4:
                self.kp[motor_name] = 100.0  # Default stiffness
                self.kd[motor_name] = 18.0   # Default damping
                self.torque_limit[motor_name] = 6.0
            elif motor_id == 6:
                self.kp[motor_name] = 28.0  # Default stiffness
                self.kd[motor_name] = 6.0   # Default damping
                self.torque_limit[motor_name] = 3.0
            elif motor_id == 8:
                self.kp[motor_name] = 28.0  # Default stiffness
                self.kd[motor_name] = 5.0   # Default damping
                self.torque_limit[motor_name] = 4.0
            elif motor_id == 10:
                self.kp[motor_name] = 28.0  # Default stiffness
                self.kd[motor_name] = 6.0   # Default damping
                self.torque_limit[motor_name] = 4.0
        
        # Control loop frequency
        self.control_frequency = 200.0  # Hz
        self.rate_limiter = RateLimiter(self.control_frequency)
        
        # Current state - store per motor
        self.measured_positions = {}
        self.measured_velocities = {}
        self.target_positions = {}  # Will be set to 1.57 and 3.14 for motors 1 and 2
        self.torque_measured = {}
        self.iq_targets = {}
        self.percentage = {}
        self.start_positions = {}
        self.start_time = time.time()
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self.stop_and_exit()
    
    def connect(self):
        """Connect to CAN bus and initialize motors."""
        print(f"🔍 Connecting to CAN channel {self.channel}...")
        
        # Define motors with model mapping
        motors = {}
        for motor_id, motor_name in zip(self.motor_ids, self.motor_names):
            # Get model from mapping, default to "rs-02" if not found
            model = self.MOTOR_MODEL_MAP.get(motor_id, "rs-02")
            motors[motor_name] = Motor(id=motor_id, model=model)
        
        # Calibration parameters
        calibration = {}
        for motor_name in self.motor_names:
            calibration[motor_name] = {"direction": 1, "homing_offset": 0.0}
        
        try:
            # Create and connect bus
            self.bus = RobstrideBus(self.channel, motors, calibration)
            self.bus.connect(handshake=True)
            
            for motor_name in self.motor_names:
                old_mode = self.bus.read(motor_name, ParameterType.MODE)
                print(f"Old mode of {motor_name}: {old_mode}")
            # Enable all motors
            for motor_id, motor_name in zip(self.motor_ids, self.motor_names):
                print(f"⚡ Activating motor ID: {motor_id}...")
                self.bus.enable(motor_name)
                time.sleep(0.5)
            
            for motor_name in self.motor_names:
                new_mode_1 = self.bus.read(motor_name, ParameterType.MODE)
                print(f"New mode_1 of {motor_name}: {new_mode_1}")
            
            # Set control parameters
            print("⚙️ Setting MIT mode control parameters...")
            for motor_name in self.motor_names:
                self.bus.write(motor_name, ParameterType.POSITION_KP, self.kp[motor_name])
                time.sleep(0.1)
                self.bus.write(motor_name, ParameterType.VELOCITY_KP, self.kd[motor_name])
                time.sleep(0.1)
                self.bus.write(motor_name, ParameterType.TORQUE_LIMIT, self.torque_limit[motor_name])
                time.sleep(0.1)
            print("⚙️ Setting mode to MIT (Mode 0)...")
            for motor_name in self.motor_names:
                self.bus.write(motor_name, ParameterType.MODE, 0)
                time.sleep(0.1)
                new_mode_2 = self.bus.read(motor_name, ParameterType.MODE)
                print(f"New mode_2 of {motor_name}: {new_mode_2}")
            time.sleep(0.2)

            for motor_name in self.motor_names:
                if new_mode_2 != 0:
                    print(f"❌ Mode change failed! Motor {motor_name} is still in Mode {new_mode_2}")
                    self.bus.disable(motor_name)
                    time.sleep(0.1)
                    self.bus.write(motor_name, ParameterType.MODE, 0)
                    time.sleep(0.1)
                    new_mode_3 = self.bus.read(motor_name, ParameterType.MODE)
                    print(f"New mode_3 of {motor_name}: {new_mode_3}")
            
            for motor_name in self.motor_names:
                self.bus.enable(motor_name)
                time.sleep(0.1)
            print("✅ Initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def set_mode(self, mode: int):
        """Switch motor operating mode.
        
        Args:
            mode: 0 = MIT Mode, 1 = Position Mode, 2 = Speed Mode
        """
        if self.bus is None:
            print("❌ Bus not connected")
            return
        
        try:
            print(f"⚙️ Switching to Mode {mode}...")
            for motor_name in self.motor_names:
                self.bus.write(motor_name, ParameterType.MODE, mode)
            self.current_mode = mode
            time.sleep(0.2)
            print(f"✅ Mode switched to {mode}")
        except Exception as e:
            print(f"❌ Failed to switch mode: {e}")
    
    def write_read_pdo_2(self, motor_name: str, target_position: float, target_velocity: float = 0.0):
        """Write position command and read status (MIT Mode).
        
        Args:
            motor_name: Name of the motor
            target_position: Target position in radians
            target_velocity: Target velocity in rad/s (feedforward)
        
        Returns:
            tuple: (measured_position, measured_velocity, iq_target, torque_measured) or (None, None, None, None) on error
        """
        if self.bus is None:
            return None, None, None, None
        
        try:
            # Send MIT operation frame with per-motor kp and kd
            self.bus.write_operation_frame(
                motor_name,
                target_position,
                self.kp[motor_name],
                self.kd[motor_name],
                target_velocity,  # velocity feedforward
                self.tff               # torque feedforward
            )
            
            # Read status frame
            position, velocity, torque, temperature = self.bus.read_operation_frame(motor_name)

            iq_target = self.bus.read(motor_name, ParameterType.IQ_TARGET)
            torque_measured = torque
            
            self.measured_positions[motor_name] = position
            self.measured_velocities[motor_name] = velocity
            self.iq_targets[motor_name] = iq_target
            self.torque_measured[motor_name] = torque_measured
            
            return position, velocity, iq_target, torque_measured
            
        except Exception as e:
            if "No response from the motor" not in str(e):
                print(f"⚠️ Communication error for {motor_name}: {e}")
            return None, None, None, None
    
    def run(self):
        """Main control loop."""
        if not self.connect():
            return
        
        # Set target positions: motor 1 = 1.57 rad, motor 2 = 3.14 rad
        if len(self.motor_names) == 5:
            self.target_positions[self.motor_names[0]] = 1.57
            self.target_positions[self.motor_names[1]] = 1.57
            self.target_positions[self.motor_names[2]] = 1.57
            self.target_positions[self.motor_names[3]] = 1.57
            self.target_positions[self.motor_names[4]] = 1.57
            self.percentage[self.motor_names[0]] = 0.0
            self.percentage[self.motor_names[1]] = 0.0
            self.percentage[self.motor_names[2]] = 0.0
            self.percentage[self.motor_names[3]] = 0.0
            self.percentage[self.motor_names[4]] = 0.0
        else:
            # Fallback for single motor
            for motor_name in self.motor_names:
                self.target_positions[motor_name] = 1.57
                self.percentage[motor_name] = 0.0
        
        print("\n" + "="*60)
        print(f"Motor IDs: {', '.join(map(str, self.motor_ids))}")
        
        # Print per-motor kp and kd
        kp_str = ', '.join([f'M{id}:{self.kp[name]:.1f}' for id, name in zip(self.motor_ids, self.motor_names)])
        kd_str = ', '.join([f'M{id}:{self.kd[name]:.1f}' for id, name in zip(self.motor_ids, self.motor_names)])
        print(f"Kp (per motor): {kp_str} Nm/rad")
        print(f"Kd (per motor): {kd_str} Nm/rad/s")
        print(f"Torque Limit: {self.torque_limit} Nm")
        print(f"Target Positions: {', '.join([f'{self.target_positions[name]:.3f} rad' for name in self.motor_names])}")

        print("="*60)
        print("Press Ctrl+C to stop")
        print("-"*60)

        for motor_name in self.motor_names:
            self.start_positions[motor_name] = self.bus.read(motor_name, ParameterType.MECHANICAL_POSITION)
            print(f"Start position of {motor_name}: {self.start_positions[motor_name]}")
        
        try:
            while self.running:
                elapsed_time = time.time() - self.start_time 

                # Send commands to all motors with their respective target positions
                for motor_name in self.motor_names:
                    target_pos = self.target_positions[motor_name]
                    
                    if self.percentage[motor_name] < 1.0:
                        self.percentage[motor_name] += 0.001
                    else:
                        self.percentage[motor_name] = 1.0
                    command_pos = self.percentage[motor_name] * target_pos + (1-self.percentage[motor_name]) * self.start_positions[motor_name]    
                    # Send command and read status
                    measured_pos, measured_vel, iq_target, torque_measured = self.write_read_pdo_2(motor_name, command_pos, 0.0)
                    
                    # Display status for all motors
                    status_lines = []
                    for motor_name, motor_id in zip(self.motor_names, self.motor_ids):
                        if motor_name in self.measured_positions and self.measured_positions[motor_name] is not None:
                            measured_pos = self.measured_positions[motor_name]
                            measured_vel = self.measured_velocities[motor_name]
                            error_rad = command_pos - measured_pos
                            error_deg = math.degrees(error_rad)
                            iq_target = self.iq_targets.get(motor_name, 0.0)
                            torque_measured = self.torque_measured.get(motor_name, 0.0)
                            
                            status_lines.append(
                                f"M{motor_id}: Tgt={command_pos:5.3f} | "
                                f"Meas={measured_pos:5.3f} ({math.degrees(measured_pos):5.1f}°) | "
                                f"Vel={measured_vel:5.3f} | "
                                f"Err={error_deg:5.2f}° | "
                                f"IQ={iq_target:5.3f}A | "
                                f"Tq={torque_measured:5.3f}Nm"
                            )
                    
                    if status_lines:
                        print(" | ".join(status_lines))
                
                # Rate limiting
                self.rate_limiter.sleep()
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_and_exit()
    
    def stop_and_exit(self):
        """Stop motors and disconnect."""
        print("\n🛑 Stopping...")
        self.running = False
        
        if self.bus:
            try:
                if self.tff > 0.0:  # MIT Mode
                    # Return to zero position
                    print("🏠 Returning to zero position...")
                    for motor_name in self.motor_names:
                        self.bus.write_operation_frame(
                            motor_name,
                            0.0,
                            self.kp[motor_name],
                            self.kd[motor_name],
                            0.0,
                            0.0
                        )
                    time.sleep(1.0)
                elif self.tff == 0.0:  # MIT Mode
                    # Return to zero position
                    print("🏠 Returning to zero position...")
                    for motor_name in self.motor_names:
                        self.bus.write_operation_frame(
                            motor_name,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0
                        )
                    time.sleep(1.0)
                
                # Disable all motors
                print("🚫 Disabling motors...")
                for motor_name in self.motor_names:
                    self.bus.disable(motor_name)
                
            except Exception as e:
                print(f"⚠️ Error during stop: {e}")
            finally:
                self.bus.disconnect()
        
        print("👋 Program ended")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='RobStride Actuator Motion Control')
    parser.add_argument('motor_ids', type=int, nargs='+', help='Motor CAN IDs (1 or 5 motors)')
    parser.add_argument('-c', type=str, default='can0', help='CAN channel (default: can0)')
    
    args = parser.parse_args()
    
    # Validate motor IDs
    if len(args.motor_ids) < 1 or len(args.motor_ids) > 5:
        print("❌ Error: Please provide 1 or 5 motor IDs")
        sys.exit(1)
    
    # Create controller
    controller = MoveActuator(
        motor_ids=args.motor_ids,
        c=args.c
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, controller._signal_handler)
    signal.signal(signal.SIGTERM, controller._signal_handler)
    
    # Run control loop
    controller.run()


if __name__ == "__main__":
    main()

