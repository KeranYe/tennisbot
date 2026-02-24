#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#

import sys, tty, termios
import os
import time

sys.path.append("../thirdparty/FTServo_Python")
# sys.path.append("..")
from scservo_sdk import *                      # Uses FTServo SDK library

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)



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
vel = [1000, 1000, 3000] # velocity: 0
acc = [50, 50, 50] # acceleration: 0

while 1:
    ch = getch()
    if ch == 'q':
        print("Quit!")
        break
    elif ch == 'r':
        print("Rotate!")
    else: 
        continue

    # # somehow in this way, the velocity commands are not executed consistently. wheel 1 runs faster than wheel 2, even though they have the same velocity and acceleration parameters.  
    # for scs_id in range(1, 4):
    #     # Add servo(id)#1~3 goal position\moving speed\moving accc value to the Syncwrite parameter storage
    #     # Servo (ID1~3) sync write velocity 1000 with acceleration 50
    #     scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 0, vel[scs_id-1]*dir[scs_id-1], acc[scs_id-1])
    #     if scs_addparam_result != True:
    #         print("[ID:%03d] groupSyncWrite addparam failed" % scs_id)

    # # Syncwrite goal position
    # scs_comm_result = packetHandler.groupSyncWrite.txPacket()
    # if scs_comm_result != COMM_SUCCESS:
    #     print("%s" % packetHandler.getTxRxResult(scs_comm_result))

    # # Clear syncwrite parameter storage
    # packetHandler.groupSyncWrite.clearParam()

    # time.sleep(1);

    for scs_id in range(1, 4): 
        scs_comm_result, scs_error = packetHandler.WriteSpec(scs_id, vel[scs_id-1]*dir[scs_id-1], acc[scs_id-1])
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(scs_comm_result))
        if scs_error != 0:
            print("%s" % packetHandler.getRxPacketError(scs_error))
    
    time.sleep(0.01);

    

# stop all servos (ID1~3)
for scs_id in range(1, 4):
    # Add servo(id)#1~3 goal position\moving speed\moving accc value to the Syncwrite parameter storage
    # Servo (ID1~3) sync write velocity 1000 with acceleration 50
    scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 0, 0, 50)
    # scs_addparam_result = packetHandler.SyncWritePosEx(scs_id, 4095, 60, 50)
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
