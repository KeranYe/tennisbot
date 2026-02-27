#!/usr/bin/env python
#
# *********     Wheel Control Example      *********
# 2 threads: one for velocity read/write, one for keyboard input

import sys, tty, termios
import os
import time
import threading

sys.path.append("../thirdparty/FTServo_Python")
# sys.path.append("..")
from scservo_sdk import *                      # Uses FTServo SDK library

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def step2rot(step):
    # convert step (0~4095) to rotation (0~2pi)
    rot = step / 4095
    return rot

def step2ang(step): 
    # convert step (0~4095) to angle (rad)
    ang = step / 4095 * 2 * 3.1415
    return ang

def step2angDeg(step):
    # convert step (0~4095) to angle (deg)
    angDeg = step / 4095 * 360
    return angDeg

def rpm2stepVel(rpm): 
    # convert rpm to step velocity (0~3200)
    step_vel = int(rpm / 0.732 * 50)
    return step_vel

def stepVel2rpm(step_vel):
    # convert step velocity (0~3200) to rpm
    rpm = step_vel * 0.732 / 50
    return rpm

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
    quit()

# Set port baudrate 1000000
if portHandler.setBaudRate(1000000):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

for scs_id in range(1, 4):
    scs_comm_result, scs_error = packetHandler.WheelMode(scs_id)
    if scs_comm_result != COMM_SUCCESS:
        print("servo %s: %s" % (scs_id, packetHandler.getTxRxResult(scs_comm_result)))
    elif scs_error != 0:
        print("servo %s: %s" % (scs_id, packetHandler.getRxPacketError(scs_error)))
    else: 
        print("servo %s: Wheel mode set" % scs_id)

# parameters
dir = [-1, 1, 1] # direction: 1 for forward, -1 for reverse
vel = [3200, 3200, 0] # velocity: 0
rpm = [30, 30, 0] # velocity: 0
acc = [50, 50, 50] # acceleration: 0

stop_event = threading.Event()
run_event = threading.Event()

def velocity_worker():
    while not stop_event.is_set():
        if not run_event.is_set():
            time.sleep(0.05)
            continue

        # velocity estimate
        for scs_id in range(1, 4):
            scs_present_position, scs_present_speed, scs_comm_result, scs_error = packetHandler.ReadPosSpeed(scs_id)
            if scs_comm_result != COMM_SUCCESS:
                print(packetHandler.getTxRxResult(scs_comm_result))
            else:
                print("[ID:%03d] Pos:%f[deg] PresSpd:%f[rpm]" % (scs_id, step2angDeg(scs_present_position), stepVel2rpm(scs_present_speed)))
            if scs_error != 0:
                print(packetHandler.getRxPacketError(scs_error))

        # velocity command
        for scs_id in range(1, 4):
            # scs_comm_result, scs_error = packetHandler.WriteSpec(scs_id, vel[scs_id-1]*dir[scs_id-1], acc[scs_id-1])
            scs_comm_result, scs_error = packetHandler.WriteSpec(scs_id, rpm2stepVel(rpm[scs_id-1])*dir[scs_id-1], acc[scs_id-1])

            if scs_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(scs_comm_result))
            if scs_error != 0:
                print("%s" % packetHandler.getRxPacketError(scs_error))

        time.sleep(1.0)

def keyboard_worker():
    print("Press 'r' to start/continue rotation, 'q' to quit")
    while not stop_event.is_set():
        ch = getch()
        if ch == 'q':
            print("Quit!")
            stop_event.set()
            break
        elif ch == 'r':
            if not run_event.is_set():
                print("Rotate!")
            run_event.set()

velocity_thread = threading.Thread(target=velocity_worker)
keyboard_thread = threading.Thread(target=keyboard_worker)

velocity_thread.start()
keyboard_thread.start()

keyboard_thread.join()
stop_event.set()
velocity_thread.join()

    

# stop all servos (ID1~3)
for scs_id in range(1, 4):
    # Add servo(id)#1~3 goal position\moving speed\moving accc value to the Syncwrite parameter storage
    # Servo (ID1~3) sync write velocity 1000 with acceleration 50
    scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 0, 0, 50)
    if scs_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

# Syncwrite goal position
scs_comm_result = packetHandler.groupSyncWrite.txPacket()
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))

# Clear syncwrite parameter storage
packetHandler.groupSyncWrite.clearParam()

time.sleep(1);
    
# Close port
portHandler.closePort()
