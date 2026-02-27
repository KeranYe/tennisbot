import sys
import tty
import termios
import math


class KeyboardInputController:
    def __init__(self, chassis):
        self.chassis = chassis

    def _getch(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _getkey(self):
        ch = self._getch()
        if ch != '\x1b':
            return ch

        seq_1 = self._getch()
        seq_2 = self._getch()
        if seq_1 != '[':
            return ch

        if seq_2 == 'A':
            return 'UP'
        if seq_2 == 'B':
            return 'DOWN'
        if seq_2 == 'C':
            return 'RIGHT'
        if seq_2 == 'D':
            return 'LEFT'
        return ch

    def worker(self):
        ang_step = math.radians(1.0)

        print("Keyboard control:")
        print("  r = run")
        print("  s = stop")
        print("  q = quit")
        print("  <space> = inlet wheel on/off (0.1 m/s)")
        print("  Up/Down = linear velocity +/- 0.01 m/s")
        print("  Left/Right = angular velocity +/- 1 deg/s")

        while not self.chassis.stop_event.is_set():
            key = self._getkey()
            if key == 'q':
                print("Quit!")
                self.chassis.run_event.clear()
                with self.chassis.state_lock:
                    self.chassis.linear_vel_cmd = 0.0
                    self.chassis.angular_vel_cmd = 0.0
                    self.chassis.inlet_enabled = False
                self.chassis.stop_event.set()
                break
            if key == 'r':
                self.chassis.run_event.set()
                continue

            if not self.chassis.run_event.is_set():
                continue

            if key == 's':
                self.chassis.run_event.clear()
                with self.chassis.state_lock:
                    self.chassis.linear_vel_cmd = 0.0
                    self.chassis.angular_vel_cmd = 0.0
                    self.chassis.inlet_enabled = False
            elif key == ' ':
                with self.chassis.state_lock:
                    self.chassis.inlet_enabled = not self.chassis.inlet_enabled
            elif key == 'UP':
                with self.chassis.state_lock:
                    self.chassis.linear_vel_cmd = self.chassis._clamp(
                        self.chassis.linear_vel_cmd + 0.01,
                        -self.chassis.max_linear_vel,
                        self.chassis.max_linear_vel,
                    )
            elif key == 'DOWN':
                with self.chassis.state_lock:
                    self.chassis.linear_vel_cmd = self.chassis._clamp(
                        self.chassis.linear_vel_cmd - 0.01,
                        -self.chassis.max_linear_vel,
                        self.chassis.max_linear_vel,
                    )
            elif key == 'LEFT':
                with self.chassis.state_lock:
                    max_angular_vel_rad = math.radians(self.chassis.max_angular_vel_deg)
                    self.chassis.angular_vel_cmd = self.chassis._clamp(
                        self.chassis.angular_vel_cmd + ang_step,
                        -max_angular_vel_rad,
                        max_angular_vel_rad,
                    )
            elif key == 'RIGHT':
                with self.chassis.state_lock:
                    max_angular_vel_rad = math.radians(self.chassis.max_angular_vel_deg)
                    self.chassis.angular_vel_cmd = self.chassis._clamp(
                        self.chassis.angular_vel_cmd - ang_step,
                        -max_angular_vel_rad,
                        max_angular_vel_rad,
                    )
            else:
                continue
