#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobStride MIT 模式位置控制 (最简可靠版)
模式: Mode 0 (MIT Mode)
通信: 循环调用 write_operation_frame

用法: python3 position_control_mit.py <motor_id>
"""

import sys
import os
import time
import math
import struct
import threading
import signal
from typing import Optional

# 尝试导入 SDK
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from robstride_dynamics import RobstrideBus, Motor, ParameterType, CommunicationType
except ImportError:
    # 假设当前目录结构
    try:
        from bus import RobstrideBus, Motor
        from protocol import ParameterType, CommunicationType
    except ImportError as e:
        print(f"❌ 无法导入 SDK: {e}")
        sys.exit(1)

class PositionControllerMIT:
    def __init__(self, motor_id: int, channel='can0'):
        self.motor_id = motor_id
        self.motor_name = f"motor_{motor_id}"
        self.channel = channel
        
        self.bus: Optional[RobstrideBus] = None
        self.lock = threading.Lock()  # 互斥锁，防止Socket冲突
        
        self.running = True
        self.connected = False
        self.target_position = 0.0  # 目标位置 (rad)
        
        # 默认参数 (MIT 模式)
        self.kp = 30.0  # 刚度 (Nm/rad)
        self.kd = 0.5   # 阻尼 (Nm/rad/s)

    def _signal_handler(self, signum, frame):
        self.stop_and_exit()

    def _set_mode_raw(self, mode: int):
        """
        使用原始 transmit 发送模式切换指令，不等待回包 (避免connect超时)
        """
        print(f"⚙️ 切换模式 (Mode {mode}) - [Raw Transmit]")
        device_id = self.bus.motors[self.motor_name].id
        param_id, param_dtype, _ = ParameterType.MODE

        # MODE 是 int8
        value_buffer = struct.pack("<bBH", mode, 0, 0)
        data = struct.pack("<HH", param_id, 0x00) + value_buffer

        self.bus.transmit(CommunicationType.WRITE_PARAMETER, self.bus.host_id, device_id, data)
        time.sleep(0.1) # 等待电机切换模式
        print(f"✅ 模式切换指令已发送")

    def connect(self):
        print(f"🔍 正在连接 CAN 通道 {self.channel}...")
        
        # 定义电机
        motors = {
            self.motor_name: Motor(id=self.motor_id, model="rs-03") # 模型型号请根据实际修改
        }
        
        # 简单的校准参数
        calibration = {
            self.motor_name: {"direction": 1, "homing_offset": 0.0}
        }

        try:
            self.bus = RobstrideBus(self.channel, motors, calibration)
            self.bus.connect(handshake=True)
            
            with self.lock:
                # 激活电机
                print(f"⚡ 激活电机 ID: {self.motor_id} ...")
                self.bus.enable(self.motor_name)
                time.sleep(0.5)

                # *********************
                # *** 核心逻辑 ***
                # *********************
                # 1. 切换到 MIT 模式 (Mode 0)
                self._set_mode_raw(0)
                
                # 2. 设置一个已知的、安全的初始目标
                print("🏠 设置初始目标为 0.0 ...")
                self.target_position = 0.0 # 设为 0 弧度
                
                # 3. 发送第一帧 MIT 指令来保持位置
                self.bus.write_operation_frame(
                    self.motor_name,
                    self.target_position,
                    self.kp,
                    self.kd,
                    0.0, # velocity_ff
                    0.0  # torque_ff
                )
                print(f"🏠 初始目标已设为: 0.0°")
            
            self.connected = True
            
            # 启动后台控制线程
            self.control_thread = threading.Thread(target=self.loop, daemon=True)
            self.control_thread.start()
            
            print("✅ 初始化完成 (Mode 0)！")
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            self.connected = False
            return False

    def loop(self):
        """控制线程：持续发送 MIT 帧以保持位置"""
        print("🔄 控制循环已启动 (Mode 0 @ 50Hz)")
        
        while self.running and self.connected:
            try:
                with self.lock:
                    # 1. 发送 MIT 帧 (只发)
                    self.bus.write_operation_frame(
                        self.motor_name,
                        self.target_position,
                        self.kp,
                        self.kd,
                        0.0, # velocity_ff
                        0.0  # torque_ff
                    )
                    
                    # 2. 读取状态帧 (只收)
                    # 这一步至关重要，用于清空 CAN 接收缓冲区，防止溢出
                    # 我们可以忽略返回值，因为我们只关心 "清空" 这个动作
                    self.bus.read_operation_frame(self.motor_name)
                    
                time.sleep(0.02) # 50Hz 控制频率
                
            except Exception as e:
                # 忽略超时，因为这在 read_operation_frame 中很常见
                if "No response from the motor" not in str(e):
                    print(f"⚠️ 通信错误: {e}")
                time.sleep(0.5)

    def set_angle(self, angle_degrees: float):
        """设置目标角度（单位：度）"""
        # 限制范围，例如 +/- 2 圈
        angle_degrees = max(-720.0, min(720.0, angle_degrees))
        # target_position 是线程安全的 (原子操作)
        self.target_position = math.radians(angle_degrees)
        print(f" -> 目标设定: {angle_degrees:.1f}°")

    def set_kp(self, kp: float):
        """设置刚度"""
        if 0 <= kp <= 500:
            self.kp = kp
            print(f" -> 刚度(Kp)设定: {self.kp:.1f}")
        else:
            print("❌ Kp 范围必须在 0-500")

    def set_kd(self, kd: float):
        """设置阻尼"""
        if 0 <= kd <= 5:
            self.kd = kd
            print(f" -> 阻尼(Kd)设定: {self.kd:.1f}")
        else:
            print("❌ Kd 范围必须在 0-5")

    def stop_and_exit(self):
        print("\n🛑 正在停止...")
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=0.5) # 等待线程退出
        
        if self.bus and self.connected:
            try:
                with self.lock:
                    # 回到零位
                    print("🏠 回到零位...")
                    self.bus.write_operation_frame(self.motor_name, 0.0, self.kp, self.kd, 0.0, 0.0)
                    time.sleep(1.0) # 等待电机移动
                    # 禁用
                    print("🚫 禁用电机...")
                    self.bus.disable(self.motor_name)
            except Exception as e:
                print(f"⚠️ 停止时出错: {e}")
            finally:
                self.bus.disconnect()
        
        print("👋 程序结束")
        sys.exit(0)

    def run_interactive(self):
        print("\n" + "="*40)
        print(f"🎮 MIT 位置控制台 (ID: {self.motor_id})")
        print("="*40)
        print("👉 直接输入数字 (单位: 度) 回车即可改变位置")
        print("👉 'kp <值>' (例如: kp 20) 来调节刚度 (消除抖动)")
        print("👉 'kd <值>' (例如: kd 0.8) 来调节阻尼 (消除抖动)")
        print("👉 '0' 或 'home' 回到零点")
        print("👉 'q' 退出")
        print(f"⚠️  当前 Kp={self.kp} | Kd={self.kd}")
        print("-" * 40)

        while True:
            try:
                cmd = input(f"[{math.degrees(self.target_position):.1f}°] >> ").strip().lower()
                
                if not cmd:
                    continue
                    
                if cmd in ['q', 'quit', 'exit']:
                    break
                
                if cmd in ['0', 'home']:
                    self.set_angle(0.0)
                    continue

                if cmd.startswith("kp "):
                    try:
                        new_kp = float(cmd.split()[1])
                        self.set_kp(new_kp)
                    except Exception:
                        print("❌ 无效 Kp. 示例: kp 20.0")
                    continue

                if cmd.startswith("kd "):
                    try:
                        new_kd = float(cmd.split()[1])
                        self.set_kd(new_kd)
                    except Exception:
                        print("❌ 无效 Kd. 示例: kd 0.5")
                    continue

                try:
                    angle = float(cmd)
                    self.set_angle(angle)
                except ValueError:
                    print("❌ 无效输入，请输入数字 (角度) 或 'kp', 'kd'")

            except KeyboardInterrupt:
                break
        
        self.stop_and_exit()

def main():
    if len(sys.argv) < 2:
        print("用法: python3 position_control_mit.py <motor_id>")
        sys.exit(1)
        
    motor_id = int(sys.argv[1])
    
    controller = PositionControllerMIT(motor_id)
    signal.signal(signal.SIGINT, controller._signal_handler)
    signal.signal(signal.SIGTERM, controller._signal_handler)
    
    if controller.connect():
        controller.run_interactive()

if __name__ == "__main__":
    main()