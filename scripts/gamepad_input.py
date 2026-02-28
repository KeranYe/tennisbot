#!/usr/bin/env python3
import sys
import time
import math
from typing import Optional
from evdev import InputDevice, ecodes


class GamepadInputController:
    """Gamepad input controller for chassis control."""
    
    # 8BitDo SN30 Pro mapping
    SN30PRO_DEVICE_NAME = "8Bitdo SF30 Pro"
    SN30PRO_BTN_A = 305
    SN30PRO_BTN_B = 304
    SN30PRO_BTN_X = 307
    SN30PRO_BTN_Y = 306
    SN30PRO_AXIS_LEFT_X = 0
    SN30PRO_AXIS_LEFT_Y = 1
    SN30PRO_AXIS_RIGHT_X = 3
    SN30PRO_AXIS_RIGHT_Y = 4
    SN30PRO_AXIS_MAX = 65535
    SN30PRO_AXIS_MIN = 0
    
    # Logitech F710 mapping
    F710_DEVICE_NAME = "Logitech Gamepad F710"
    F710_BTN_A = ecodes.BTN_A
    F710_BTN_B = ecodes.BTN_B
    F710_BTN_X = ecodes.BTN_X
    F710_BTN_Y = ecodes.BTN_Y
    F710_AXIS_LEFT_X = ecodes.ABS_HAT0X
    F710_AXIS_LEFT_Y = ecodes.ABS_HAT0Y
    F710_AXIS_RIGHT_X = ecodes.ABS_RX
    F710_AXIS_RIGHT_Y = ecodes.ABS_RY
    F710_AXIS_MAX = 32767
    F710_AXIS_MIN = -32768

    def __init__(self, chassis):
        self.chassis = chassis
        self.device = None
        self.gamepad_type = None
        
        # Button mappings (will be set based on detected gamepad)
        self.btn_run = None
        self.btn_stop = None
        self.btn_inlet_toggle = None
        self.btn_quit = None
        
        # Axis mappings
        self.axis_linear = None
        self.axis_angular = None
        self.axis_max = None
        self.axis_min = None
        
        # State
        self.linear_vel_raw = 0.0
        self.angular_vel_raw = 0.0
        self.deadzone = 0.15  # 15% deadzone for stick drift

    def _find_gamepad(self) -> Optional[InputDevice]:
        """Find and connect to a supported gamepad."""
        candidates = [
            "/dev/input/event0",
            "/dev/input/event1",
            "/dev/input/event2",
            "/dev/input/event3",
            "/dev/input/event4",
            "/dev/input/event28",
            "/dev/input/event29",
        ]
        
        # Try to find 8BitDo SN30 Pro
        for path in candidates:
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            
            if dev.name and self.SN30PRO_DEVICE_NAME in dev.name:
                print(f"Connected to gamepad: {dev.name}")
                print(f"Using device: {path}")
                self.gamepad_type = "SN30Pro"
                self._setup_sn30pro_mapping()
                return dev
            
            dev.close()
        
        # Try to find Logitech F710
        for path in candidates:
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            
            if dev.name and self.F710_DEVICE_NAME in dev.name:
                print(f"Connected to gamepad: {dev.name}")
                print(f"Using device: {path}")
                self.gamepad_type = "F710"
                self._setup_f710_mapping()
                return dev
            
            dev.close()
        
        return None

    def _setup_sn30pro_mapping(self):
        """Setup button and axis mappings for 8BitDo SN30 Pro."""
        self.btn_run = self.SN30PRO_BTN_X
        self.btn_stop = self.SN30PRO_BTN_Y
        self.btn_inlet_toggle = self.SN30PRO_BTN_A
        self.btn_quit = self.SN30PRO_BTN_B
        
        self.axis_linear = self.SN30PRO_AXIS_RIGHT_Y
        self.axis_angular = self.SN30PRO_AXIS_LEFT_X
        self.axis_max = self.SN30PRO_AXIS_MAX
        self.axis_min = self.SN30PRO_AXIS_MIN

    def _setup_f710_mapping(self):
        """Setup button and axis mappings for Logitech F710."""
        self.btn_run = self.F710_BTN_X
        self.btn_stop = self.F710_BTN_Y
        self.btn_inlet_toggle = self.F710_BTN_A
        self.btn_quit = self.F710_BTN_B
        
        self.axis_linear = self.F710_AXIS_RIGHT_Y
        self.axis_angular = self.F710_AXIS_LEFT_X
        self.axis_max = self.F710_AXIS_MAX
        self.axis_min = self.F710_AXIS_MIN

    def _normalize_axis(self, value: int, invert: bool = False) -> float:
        """
        Normalize axis value to [-1, 1] range with deadzone.
        
        Args:
            value: Raw axis value
            invert: If True, invert the output direction
        
        Returns:
            Normalized value in range [-1, 1]
        """
        # Normalize to [0, 1] or [-1, 1] depending on axis range
        if self.axis_min >= 0:
            # Unsigned axis (0 to axis_max)
            normalized = (value - self.axis_max / 2) / (self.axis_max / 2)
        else:
            # Signed axis (axis_min to axis_max)
            normalized = value / self.axis_max
        
        # Apply deadzone
        if abs(normalized) < self.deadzone:
            normalized = 0.0
        else:
            # Scale to compensate for deadzone
            sign = 1.0 if normalized > 0 else -1.0
            normalized = sign * (abs(normalized) - self.deadzone) / (1.0 - self.deadzone)
        
        # Clamp to [-1, 1]
        normalized = max(-1.0, min(1.0, normalized))
        
        # Invert if needed
        if invert:
            normalized = -normalized
        
        return normalized

    def _handle_button_event(self, code: int, value: int):
        """Handle button press/release events."""
        if value == 1:  # Button pressed
            if code == self.btn_run:
                print("Button X pressed: RUN enabled")
                self.chassis.run_event.set()
                
            elif code == self.btn_stop:
                print("Button Y pressed: STOP (RUN disabled)")
                self.chassis.run_event.clear()
                with self.chassis.state_lock:
                    self.chassis.linear_vel_cmd = 0.0
                    self.chassis.angular_vel_cmd = 0.0
                    self.chassis.inlet_enabled = False
                
            elif code == self.btn_inlet_toggle:
                with self.chassis.state_lock:
                    self.chassis.inlet_enabled = not self.chassis.inlet_enabled
                    state = "ON" if self.chassis.inlet_enabled else "OFF"
                print(f"Button A pressed: Inlet {state}")
                
            elif code == self.btn_quit:
                print("Button B pressed: QUIT")
                return True  # Signal to quit
        
        return False

    def _handle_axis_event(self, code: int, value: int):
        """Handle axis movement events."""
        if code == self.axis_linear:
            # Right stick Y-axis: backward (-1) to forward (+1)
            # Invert because typically down is positive in raw values
            self.linear_vel_raw = self._normalize_axis(value, invert=True)
            linear_vel_cmd = self.linear_vel_raw * self.chassis.max_linear_vel
            
            with self.chassis.state_lock:
                self.chassis.linear_vel_cmd = linear_vel_cmd
                
        elif code == self.axis_angular:
            # Left stick X-axis: left (-1) to right (+1)
            # For angular velocity: right stick should turn right (positive angular vel)
            self.angular_vel_raw = self._normalize_axis(value, invert=False)
            max_angular_vel_rad = math.radians(self.chassis.max_angular_vel_deg)
            angular_vel_cmd = -self.angular_vel_raw * max_angular_vel_rad  # Invert for intuitive control
            
            with self.chassis.state_lock:
                self.chassis.angular_vel_cmd = angular_vel_cmd

    def worker(self):
        """Main worker thread for gamepad input processing."""
        print("\n=== Gamepad Input Controller ===")
        print("Searching for supported gamepad...")
        
        self.device = self._find_gamepad()
        if self.device is None:
            print(
                "No supported gamepad found.\n"
                "Supported gamepads: 8Bitdo SF30 Pro, Logitech Gamepad F710\n"
                "Please connect a gamepad and try again.",
                file=sys.stderr,
            )
            return
        
        try:
            self.device.grab()
        except OSError:
            print("Warning: Could not grab device exclusively", file=sys.stderr)
        
        print(f"\nGamepad Type: {self.gamepad_type}")
        print("\nControls:")
        print("  Right Stick Y-axis : Linear velocity (forward/backward)")
        print("  Left Stick X-axis  : Angular velocity (turn left/right)")
        print("  Button X           : Enable RUN")
        print("  Button Y           : STOP (disable RUN)")
        print("  Button A           : Toggle inlet wheel")
        print("  Button B           : QUIT")
        print("\nGamepad connected. Ready for input...\n")
        
        try:
            while True:
                # Read events (blocking)
                for event in self.device.read_loop():
                    if event.type == ecodes.EV_KEY:
                        # Button event
                        should_quit = self._handle_button_event(event.code, event.value)
                        if should_quit:
                            return
                            
                    elif event.type == ecodes.EV_ABS:
                        # Axis event
                        self._handle_axis_event(event.code, event.value)
                        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
        finally:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self.device.close()
            print("Gamepad disconnected")


def main():
    """Test function for standalone execution."""
    print("GamepadInputController test mode")
    print("This module is designed to be used with chassis_control.py")
    print("Run chassis_control.py and modify it to use startGamepadControl()")
    return 0


if __name__ == "__main__":
    sys.exit(main())