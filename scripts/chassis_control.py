#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#

import sys
import os
import time
import math
import threading

sys.path.append("../thirdparty/FTServo_Python")
# sys.path.append("..")
from scservo_sdk import *                      # Uses FTServo SDK library
from keyboad_input import KeyboardInputController

# define wheel class
class Wheel:
    # member variables: parameters 
    scs_id = 0
    dir = 1 # direction: 1 for forward, -1 for reverse
    dia = 0.1 # diameter of the wheel: m
    reduction = 1 # gear reduction ratio

    # interface parameters
    packetHandler = None
    
    # member variables: states
    ang_vel_cmd = 0 # angular velocity command: rad/s
    ang_acc_cmd = 0 # acceleration: rad/s^2
    ang_pos_est = 0 # angular position estimate: rad
    ang_vel_est = 0 # angular velocity estimate: rad/s
    # ang_acc_est = 0 # acceleration estimate: rad/s^2

    # member functions: major interface 
    def __init__(self, packetHandler, scs_id, dir, dia, reduction=1):
        self.packetHandler = packetHandler
        self.scs_id = scs_id
        self.dir = dir
        self.dia = dia
        self.reduction = reduction

        # set wheel mode
        scs_comm_result, scs_error = self.packetHandler.WheelMode(self.scs_id)
        if scs_comm_result != COMM_SUCCESS:
            print("servo %s: %s" % (self.scs_id, self.packetHandler.getTxRxResult(scs_comm_result)))
        elif scs_error != 0:
            print("servo %s: %s" % (self.scs_id, self.packetHandler.getRxPacketError(scs_error)))
        else: 
            print("servo %s: Wheel mode set" % self.scs_id)

    def stop(self):
        scs_comm_result, scs_error = self.packetHandler.WriteSpec(self.scs_id, 0, 50)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
        if scs_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(scs_error))

    def setCommands(self, ang_vel_cmd, ang_acc_cmd):
        self.ang_vel_cmd = ang_vel_cmd
        self.ang_acc_cmd = ang_acc_cmd

        step_vel_cmd = self.angVel2stepVel(ang_vel_cmd)
        step_acc_cmd = self.angVel2stepVel(ang_acc_cmd)

        speed_cmd = int(step_vel_cmd * self.dir * self.reduction)
        acc_cmd = int(step_acc_cmd)

        scs_comm_result, scs_error = self.packetHandler.WriteSpec(self.scs_id, speed_cmd, acc_cmd)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
        if scs_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(scs_error))
    
    def getMeasurements(self):
        scs_present_position, scs_present_speed, scs_comm_result, scs_error = self.packetHandler.ReadPosSpeed(self.scs_id)
        if scs_comm_result != COMM_SUCCESS:
            print(self.packetHandler.getTxRxResult(scs_comm_result))
        else:
            self.ang_pos_est = self.step2angRad(scs_present_position) * self.dir / self.reduction
            self.ang_vel_est = self.stepVel2angVel(scs_present_speed) * self.dir / self.reduction
            # print("[ID:%03d] PresPos:%f PresSpd:%f" % (self.scs_id, self.ang_pos_est, self.ang_vel_est))
        if scs_error != 0:
            print(self. packetHandler.getRxPacketError(scs_error))

    # member functions: helpers
    def step2angDeg(self, step):
        # convert step position (0~4095) to angular position (deg)
        ang_deg = step / 4095 * 360
        return ang_deg
    
    def step2angRad(self, step):
        # convert step position (0~4095) to angular position (rad)
        ang_rad = step / 4095 * 2 * 3.1415
        return ang_rad
    
    def angDeg2step(self, ang_deg):
        # convert angular position (deg) to step position (0~4095)
        step = int(ang_deg / 360 * 4095)
        return step
    
    def angRad2step(self, ang_rad):
        # convert angular position (rad) to step position (0~4095)
        step = int(ang_rad / (2 * 3.1415) * 4095)
        return step

    def angVel2stepVel(self, ang_vel):
        # convert angular velocity (rad/s) to step velocity (0~3200)
        step_vel = int(ang_vel * 9.549 / 0.732 * 50)
        return step_vel
    
    def rpm2stepVel(self, rpm):
        # convert rpm to step velocity (0~3200)
        step_vel = int(rpm / 0.732 * 50)
        return step_vel
    
    def stepVel2angVel(self, step_vel):
        # convert step velocity (0~3200) to angular velocity (rad/s)
        ang_vel = step_vel * 0.732 / 50 / 9.549
        return ang_vel
    
    def stepVel2rpm(self, step_vel):
        # convert step velocity (0~3200) to rpm
        rpm = step_vel * 0.732 / 50
        return rpm

class Chassis:
    # member variables: parameters
    left_wheel = None
    right_wheel = None
    inlet_wheel = None

    shaft_distance = 0.2 # distance between left and right wheel: m
    wheel_diameter = 0.1 # diameter of the wheel: m
    inlet_disk_diameter = 0.05 # diameter of the inlet wheel: m

    wheel_reduction = 1 # gear reduction ratio
    inlet_reduction = 0.1 # gear reduction ratio of the inlet wheel

    max_linear_vel = 0.25 # maximum linear velocity: m/s
    max_angular_vel_deg = 45.0 # maximum angular velocity: deg/s
    max_inlet_vel = 2.0 # maximum inlet velocity: m/s

    # member variables: interface parameters
    packetHandler = None

    # member variables: states
    linear_vel_cmd = 0 # linear velocity command: m/s
    angular_vel_cmd = 0 # angular velocity command: rad/s
    inlet_vel_cmd = 0 # inlet wheel velocity command: m/s

    linear_vel_est = 0 # linear velocity estimate: m/s
    angular_vel_est = 0 # angular velocity estimate: rad/s
    inlet_vel_est = 0 # inlet wheel velocity estimate: m/s

    # member functions: major interface
    def __init__(self, packetHandler, shaft_distance, wheel_diameter, inlet_disk_diameter, 
                 wheel_reduction, inlet_reduction, 
                 max_linear_vel, max_angular_vel_deg, max_inlet_vel ):
        self.packetHandler = packetHandler
        self.shaft_distance = shaft_distance
        self.wheel_diameter = wheel_diameter
        self.inlet_disk_diameter = inlet_disk_diameter
        self.wheel_reduction = wheel_reduction
        self.inlet_reduction = inlet_reduction
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel_deg = max_angular_vel_deg
        self.max_inlet_vel = max_inlet_vel

        self.left_wheel = Wheel(packetHandler, scs_id=1, dir=-1, dia=self.wheel_diameter, reduction=self.wheel_reduction)
        self.right_wheel = Wheel(packetHandler, scs_id=2, dir=1, dia=self.wheel_diameter, reduction=self.wheel_reduction)
        self.inlet_wheel = Wheel(packetHandler, scs_id=3, dir=1, dia=self.inlet_disk_diameter, reduction=self.inlet_reduction)

        self.stop_event = threading.Event()
        self.run_event = threading.Event()
        self.state_lock = threading.Lock()
        self.inlet_enabled = False
        self.inlet_vel_cmd = self.max_inlet_vel
        self.keyboard_controller = KeyboardInputController(self)

    def _clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def setMotionCommands(self, linear_vel_cmd, angular_vel_cmd):
        linear_vel_cmd = self._clamp(linear_vel_cmd, -self.max_linear_vel, self.max_linear_vel)
        max_angular_vel_rad = math.radians(self.max_angular_vel_deg)
        angular_vel_cmd = self._clamp(angular_vel_cmd, -max_angular_vel_rad, max_angular_vel_rad)

        self.linear_vel_cmd = linear_vel_cmd
        self.angular_vel_cmd = angular_vel_cmd

        left_wheel_ang_vel_cmd = (linear_vel_cmd - angular_vel_cmd * self.shaft_distance / 2) / (self.wheel_diameter / 2)
        right_wheel_ang_vel_cmd = (linear_vel_cmd + angular_vel_cmd * self.shaft_distance / 2) / (self.wheel_diameter / 2)

        self.left_wheel.setCommands(left_wheel_ang_vel_cmd, ang_acc_cmd=0)
        self.right_wheel.setCommands(right_wheel_ang_vel_cmd, ang_acc_cmd=0)

    def getMotionMeasurements(self):
        self.left_wheel.getMeasurements()
        self.right_wheel.getMeasurements()

        left_wheel_ang_vel_est = self.left_wheel.ang_vel_est
        right_wheel_ang_vel_est = self.right_wheel.ang_vel_est

        self.linear_vel_est = (left_wheel_ang_vel_est + right_wheel_ang_vel_est) * (self.wheel_diameter / 2) / 2
        self.angular_vel_est = (right_wheel_ang_vel_est - left_wheel_ang_vel_est) * (self.wheel_diameter / 2) / self.shaft_distance

        # print("Linear Vel Est: %f m/s, Angular Vel Est: %f rad/s" % (self.linear_vel_est, self.angular_vel_est))

    def stop(self):
        with self.state_lock:
            self.linear_vel_cmd = 0.0
            self.angular_vel_cmd = 0.0
            self.inlet_enabled = False
        self.left_wheel.stop()
        self.right_wheel.stop()
        self.inlet_wheel.stop()
    
    def setInletWheelCommand(self, ang_vel_cmd, ang_acc_cmd):
        self.inlet_wheel.setCommands(ang_vel_cmd, ang_acc_cmd)

    def getInletWheelMeasurement(self):
        self.inlet_wheel.getMeasurements()
        self.inlet_vel_est = self.inlet_wheel.ang_vel_est * (self.inlet_disk_diameter / 2)

    def motionControlWorker(self):
        control_period_s = 0.1  # 10 Hz
        print_period_s = 1.0    # 1 Hz
        next_print_time = time.monotonic() + print_period_s

        while not self.stop_event.is_set():
            loop_start = time.monotonic()

            with self.state_lock:
                linear_vel_cmd = self.linear_vel_cmd
                angular_vel_cmd = self.angular_vel_cmd
                inlet_enabled = self.inlet_enabled
                inlet_vel_cmd = self._clamp(self.inlet_vel_cmd, 0.0, self.max_inlet_vel)
            run_enabled = self.run_event.is_set()

            if run_enabled:
                self.setMotionCommands(linear_vel_cmd, angular_vel_cmd)
            else:
                self.setMotionCommands(0.0, 0.0)

            inlet_active = run_enabled and inlet_enabled
            if inlet_active:
                inlet_ang_vel_cmd = inlet_vel_cmd / (self.inlet_disk_diameter / 2)
                self.setInletWheelCommand(inlet_ang_vel_cmd, 0)
            else:
                self.setInletWheelCommand(0, 0)

            self.getMotionMeasurements()
            self.getInletWheelMeasurement()

            now = time.monotonic()
            if now >= next_print_time:
                print(
                    "run=%s | cmd: linear=%.2f m/s angular=%.2f deg/s inlet=%.2f m/s(%s) | est: linear=%.2f m/s angular=%.2f deg/s inlet=%.2f m/s"
                    % (
                        "on" if run_enabled else "off",
                        linear_vel_cmd,
                        math.degrees(angular_vel_cmd),
                        inlet_vel_cmd if inlet_active else 0.0,
                        "on" if inlet_active else "off",
                        self.linear_vel_est,
                        math.degrees(self.angular_vel_est),
                        self.inlet_vel_est,
                    )
                )
                next_print_time = now + print_period_s

            elapsed = time.monotonic() - loop_start
            sleep_time = control_period_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.stop()

    def _startInputControl(self, input_worker):
        input_thread = threading.Thread(target=input_worker)
        motion_thread = threading.Thread(target=self.motionControlWorker)

        motion_thread.start()
        input_thread.start()

        input_thread.join()
        self.stop_event.set()
        motion_thread.join()

    def startKeyboardControl(self):
        self._startInputControl(self.keyboard_controller.worker)

def main():
    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    portHandler = PortHandler('/dev/ttyACM0')# ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    # Initialize PacketHandler instance
    # Get methods and members of Protocol
    packetHandler = sms_sts(portHandler)
        
    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        return

    # Set port baudrate 1000000
    if portHandler.setBaudRate(1000000):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        portHandler.closePort()
        return

    chassis = Chassis(packetHandler, 
                    shaft_distance=0.2, wheel_diameter=0.1, inlet_disk_diameter=0.05, 
                    wheel_reduction=1, inlet_reduction=0.1, 
                    max_linear_vel=0.2, max_angular_vel_deg=45, max_inlet_vel=3.0
                    )

    chassis.startKeyboardControl()
        
    # Close port
    portHandler.closePort()


if __name__ == '__main__':
    main()
