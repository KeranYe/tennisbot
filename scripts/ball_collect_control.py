#!/usr/bin/env python3
"""
Ball collection controller using visual servoing with IMX500 camera.
Controls chassis to track trajectory from inlet frame to ball position.
"""

import sys
import os
import time
import math
import threading
import argparse
from pathlib import Path
import cv2
import numpy as np

sys.path.append("../thirdparty/FTServo_Python")
from scservo_sdk import *
from picamera2 import Picamera2, MappedArray

try:
    from picamera2.devices import IMX500
except Exception:
    from picamera2.devices.imx500 import IMX500

from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from ball_finder import BallFinder, load_calibration, default_coco_labels, fmt_cm
from chassis_control import Chassis

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Constants from ball_finder
CALIB_FILE = "../data/camera_intrinsics_640x480_20cmfocus.npz"
FRAME_W, FRAME_H = 640, 480
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
TARGET_LABELS = {"sports ball"}
CONF_MIN = 0.30
FAR_NCNN_MODEL_PATH = "../yolo_models/yolo26n_480_ncnn_model"
FAR_NCNN_IMG_SIZE = 480
FAR_NCNN_CONF = 0.20
FAR_CSI0_HFOV_DEG = 66.0
FAR_APPROACH_TURN_SIGN = -1.0  # Use -1 if steering appears flipped
FAR_BALL_DIAMETER_M = 0.067


def find_class_id(model, class_name):
    names = model.names
    if isinstance(names, dict):
        inv = {v: k for k, v in names.items()}
        if class_name not in inv:
            raise ValueError(f"Class '{class_name}' not found in model names")
        return int(inv[class_name])
    if class_name not in names:
        raise ValueError(f"Class '{class_name}' not found in model names")
    return int(names.index(class_name))

# Trajectory planner types
TRAJECTORY_PLANNER_BEZIER = "bezier"  # Quadratic Bézier curve (default)
TRAJECTORY_PLANNER_SHAPED = "shaped"  # Close-range approach shaping
TRAJECTORY_PLANNER_DIRECT = "direct"  # Simple direct line to ball
TRAJECTORY_PLANNER_ARC = "arc"  # Smooth arc trajectory


class InletVisualController:
    """
    Visual servoing controller for ball collection.
    Plans trajectory from inlet to ball and generates chassis control commands.
    """
    
    def __init__(
        self,
        chassis,
        ball_finder,
        T_chassis_to_camera,
        T_chassis_to_inlet,
        control_rate_hz=10.0,
        trajectory_waypoints=10,
        lookahead_steps=1,
        target_approach_distance=0.10,  # Stop 10cm from ball
        trajectory_planner=TRAJECTORY_PLANNER_BEZIER,  # Default planner
    ):
        """
        Args:
            chassis: Chassis instance for motion control
            ball_finder: BallFinder instance for ball detection
            T_chassis_to_camera: 4x4 transformation matrix from chassis center to camera frame
            T_chassis_to_inlet: 4x4 transformation matrix from chassis center to inlet frame
            control_rate_hz: Control loop frequency
            trajectory_waypoints: Maximum number of trajectory waypoints to generate
            lookahead_steps: Number of trajectory steps to execute before replanning
            target_approach_distance: Desired distance to stop from ball (meters)
            trajectory_planner: Type of trajectory planner to use (default: bezier)
        """
        self.chassis = chassis
        self.ball_finder = ball_finder
        self.T_chassis_to_camera = np.array(T_chassis_to_camera, dtype=np.float64)
        self.T_chassis_to_inlet = np.array(T_chassis_to_inlet, dtype=np.float64)
        self.control_rate_hz = float(control_rate_hz)
        self.trajectory_waypoints = int(trajectory_waypoints)
        self.lookahead_steps = int(lookahead_steps)
        self.target_approach_distance = float(target_approach_distance)
        
        self.control_period = 1.0 / self.control_rate_hz
        self.trajectory_planner = trajectory_planner
        self.ball_detected = False
        self.ball_pos_ballfinder = None  # Ball position from BallFinder frame
        self.ball_pos_chassis = None  # Ball position in chassis frame
        self.trajectory = []  # List of waypoints in chassis frame
        self.current_traj_step = 0
        
        # Trajectory planning rate (slower for visibility)
        self.trajectory_update_period = 1.0  # Update trajectory every 1 second
        self.last_trajectory_update = 0.0
        
        # Control gains
        self.k_linear = 0.5    # Proportional gain for linear velocity
        self.k_angular = 2.0   # Proportional gain for angular velocity
        self.min_linear_vel = 0.05  # Minimum linear velocity (m/s) to avoid slow approach
        self.min_angular_vel = math.radians(5.0)  # Minimum angular velocity (rad/s) to avoid slow turning
        self.waypoint_reach_threshold_m = 0.005  # 5mm, smaller than close waypoint spacing
        
    def transform_point(self, point, T):
        """Transform a 3D point using homogeneous transformation matrix."""
        if point is None:
            return None
        p_homo = np.array([point[0], point[1], point[2], 1.0])
        p_transformed = T @ p_homo
        return p_transformed[:3]
    
    def ball_camera_to_chassis(self, ball_pos_camera):
        """Convert ball position from camera frame to chassis center frame."""
        # NOTE: T_chassis_to_camera describes chassis->camera transform
        # To go from camera->chassis, we need the inverse
        # T_camera_to_chassis = np.linalg.inv(self.T_chassis_to_camera)
        # T_camera_to_chassis = self.T_chassis_to_camera; 
        result = self.transform_point(ball_pos_camera, self.T_chassis_to_camera)
        print(f"[DEBUG] ball_camera: {fmt_cm(ball_pos_camera)} -> chassis: {fmt_cm(result) if result is not None else 'None'}")
        return result
    
    def plan_trajectory(self, ball_pos_chassis):
        """
        Plan trajectory from inlet to ball position using selected planner.
        Dispatches to the appropriate planning method based on trajectory_planner type.
        
        Args:
            ball_pos_chassis: Ball position in chassis frame [x, y, z]
            
        Returns:
            List of waypoints in chassis frame, each [x, y, theta]
        """
        if ball_pos_chassis is None:
            return []
        
        # Dispatch to appropriate planner
        if self.trajectory_planner == TRAJECTORY_PLANNER_BEZIER:
            return self._plan_trajectory_bezier(ball_pos_chassis)
        elif self.trajectory_planner == TRAJECTORY_PLANNER_DIRECT:
            return self._plan_trajectory_direct(ball_pos_chassis)
        elif self.trajectory_planner == TRAJECTORY_PLANNER_ARC:
            return self._plan_trajectory_arc(ball_pos_chassis)
        elif self.trajectory_planner == TRAJECTORY_PLANNER_SHAPED:
            return self._plan_trajectory_shaped(ball_pos_chassis)
        else:
            print(f"[WARNING] Unknown planner type '{self.trajectory_planner}', using bezier planner")
            return self._plan_trajectory_bezier(ball_pos_chassis)
    
    def _plan_trajectory_bezier(self, ball_pos_chassis):
        """
        Quadratic Bézier curve trajectory from inlet to ball.
        Control point is the projection of the midpoint onto the chassis x-axis.
        
        Args:
            ball_pos_chassis: Ball position in chassis frame [x, y, z]
            
        Returns:
            List of waypoints in chassis frame, each [x, y, theta]
        """
        inlet_pos_chassis = self.T_chassis_to_inlet[:3, 3]
        ball_x, ball_y = ball_pos_chassis[0], ball_pos_chassis[1]
        inlet_x, inlet_y = inlet_pos_chassis[0], inlet_pos_chassis[1]
        
        distance_to_ball = math.hypot(ball_x - inlet_x, ball_y - inlet_y)
        
        print(f"[DEBUG] Bézier trajectory: ball={fmt_cm(ball_pos_chassis)}, "
              f"inlet={fmt_cm(inlet_pos_chassis)}, distance={distance_to_ball*100:.1f}cm")
        
        if distance_to_ball < 0.01:
            return []
        
        # Bézier control points
        P0 = np.array([inlet_x, inlet_y])  # Start: inlet
        P2 = np.array([ball_x, ball_y])    # End: ball
        
        # Middle control point: project midpoint onto chassis x-axis (y=0)
        midpoint_x = (inlet_x + ball_x) / 2.0
        midpoint_y = (inlet_y + ball_y) / 2.0
        P1 = np.array([midpoint_x, 0.0])  # Project onto x-axis
        
        print(f"[DEBUG] Bézier control points: P0={fmt_cm([P0[0], P0[1], 0])}, "
              f"P1={fmt_cm([P1[0], P1[1], 0])}, P2={fmt_cm([P2[0], P2[1], 0])}")
        
        # Generate waypoints
        min_waypoint_step_m = 0.015
        waypoints_from_distance = max(1, int(distance_to_ball / min_waypoint_step_m))
        num_waypoints = min(self.trajectory_waypoints, waypoints_from_distance)
        
        trajectory = []
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            
            # Quadratic Bézier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            one_minus_t = 1.0 - t
            wp_pos = (one_minus_t * one_minus_t * P0 + 
                     2.0 * one_minus_t * t * P1 + 
                     t * t * P2)
            
            wp_x, wp_y = wp_pos[0], wp_pos[1]
            
            # Tangent direction: derivative of Bézier curve
            # B'(t) = 2(1-t)(P₁-P₀) + 2t(P₂-P₁)
            tangent = 2.0 * one_minus_t * (P1 - P0) + 2.0 * t * (P2 - P1)
            wp_theta = math.atan2(tangent[1], tangent[0])
            
            trajectory.append([wp_x, wp_y, wp_theta])
        
        print(f"[DEBUG] Created Bézier trajectory with {len(trajectory)} waypoints")
        return trajectory
    
    def _plan_trajectory_direct(self, ball_pos_chassis):
        """
        Simple direct line trajectory from inlet to ball.
        
        Args:
            ball_pos_chassis: Ball position in chassis frame [x, y, z]
            
        Returns:
            List of waypoints in chassis frame, each [x, y, theta]
        """
        inlet_pos_chassis = self.T_chassis_to_inlet[:3, 3]
        ball_x, ball_y = ball_pos_chassis[0], ball_pos_chassis[1]
        distance_to_ball = math.hypot(ball_x - inlet_pos_chassis[0], 
                                       ball_y - inlet_pos_chassis[1])
        
        print(f"[DEBUG] Direct trajectory: ball={fmt_cm(ball_pos_chassis)}, "
              f"inlet={fmt_cm(inlet_pos_chassis)}, distance={distance_to_ball*100:.1f}cm")
        
        if distance_to_ball < 0.01:
            return []
        
        # Heading to ball
        angle_to_ball = math.atan2(ball_y - inlet_pos_chassis[1], 
                                    ball_x - inlet_pos_chassis[0])
        
        # Generate waypoints with minimum step size
        min_waypoint_step_m = 0.015
        waypoints_from_distance = max(1, int(distance_to_ball / min_waypoint_step_m))
        num_waypoints = min(self.trajectory_waypoints, waypoints_from_distance)
        
        trajectory = []
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            wp_x = inlet_pos_chassis[0] + (ball_x - inlet_pos_chassis[0]) * alpha
            wp_y = inlet_pos_chassis[1] + (ball_y - inlet_pos_chassis[1]) * alpha
            trajectory.append([wp_x, wp_y, angle_to_ball])
        
        print(f"[DEBUG] Created direct trajectory with {len(trajectory)} waypoints")
        return trajectory
    
    def _plan_trajectory_arc(self, ball_pos_chassis):
        """
        Smooth arc trajectory from inlet to ball.
        Uses a circular arc for smoother motion.
        
        Args:
            ball_pos_chassis: Ball position in chassis frame [x, y, z]
            
        Returns:
            List of waypoints in chassis frame, each [x, y, theta]
        """
        inlet_pos_chassis = self.T_chassis_to_inlet[:3, 3]
        ball_x, ball_y = ball_pos_chassis[0], ball_pos_chassis[1]
        distance_to_ball = math.hypot(ball_x - inlet_pos_chassis[0], 
                                       ball_y - inlet_pos_chassis[1])
        
        print(f"[DEBUG] Arc trajectory: ball={fmt_cm(ball_pos_chassis)}, "
              f"inlet={fmt_cm(inlet_pos_chassis)}, distance={distance_to_ball*100:.1f}cm")
        
        if distance_to_ball < 0.01:
            return []
        
        # Direction angles
        angle_to_ball = math.atan2(ball_y - inlet_pos_chassis[1], 
                                    ball_x - inlet_pos_chassis[0])
        
        # Generate waypoints with smooth arc
        min_waypoint_step_m = 0.015
        waypoints_from_distance = max(1, int(distance_to_ball / min_waypoint_step_m))
        num_waypoints = min(self.trajectory_waypoints, waypoints_from_distance)
        
        trajectory = []
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            # Use smoothstep for acceleration/deceleration
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            
            wp_x = inlet_pos_chassis[0] + (ball_x - inlet_pos_chassis[0]) * smooth_alpha
            wp_y = inlet_pos_chassis[1] + (ball_y - inlet_pos_chassis[1]) * smooth_alpha
            
            # Tangent angle along the path
            wp_theta = angle_to_ball
            trajectory.append([wp_x, wp_y, wp_theta])
        
        print(f"[DEBUG] Created arc trajectory with {len(trajectory)} waypoints")
        return trajectory
    
    def _plan_trajectory_shaped(self, ball_pos_chassis):
        """
        Shaped trajectory with close-range approach constraints.
        This is the original/default planner with special handling for nearby balls.
        
        Args:
            ball_pos_chassis: Ball position in chassis frame [x, y, z]
            
        Returns:
            List of waypoints in chassis frame, each [x, y, theta]
        """
        if ball_pos_chassis is None:
            return []
        
        # Get inlet position in chassis frame
        inlet_pos_chassis = self.T_chassis_to_inlet[:3, 3]
        
        # Target position: go to ball center
        ball_x, ball_y = ball_pos_chassis[0], ball_pos_chassis[1]
        distance_to_ball = math.hypot(ball_x - inlet_pos_chassis[0], 
                                       ball_y - inlet_pos_chassis[1])
        
        print(f"[DEBUG] Trajectory planning: ball={fmt_cm(ball_pos_chassis)}, inlet={fmt_cm(inlet_pos_chassis)}, distance={distance_to_ball*100:.1f}cm")
        
        if distance_to_ball < 0.01:  # Already at ball
            print(f"[DEBUG] Already at ball, no trajectory")
            return []
        
        # Direction vector from inlet to ball
        dx = ball_x - inlet_pos_chassis[0]
        dy = ball_y - inlet_pos_chassis[1]

        # Close-range approach shaping:
        # When ball is close to inlet, bias trajectory to be nearly perpendicular
        # to inlet baseline (i.e., reduce lateral motion, approach mostly along x).
        close_range_m = 0.18
        max_deviation_angle_deg = 15.0  # Max allowed deviation from x-axis in early segment
        max_deviation_angle = math.radians(max_deviation_angle_deg)

        target_x = ball_x
        target_y = ball_y
        close_mode = False
        blend = 1.0
        if distance_to_ball < close_range_m:
            blend = max(0.0, min(1.0, distance_to_ball / close_range_m))
            close_mode = True
            print(
                f"[DEBUG] Close ball shaping: dist={distance_to_ball*100:.1f}cm, "
                f"max_angle={max_deviation_angle_deg:.1f}deg"
            )

        angle_to_ball = math.atan2(target_y - inlet_pos_chassis[1], target_x - inlet_pos_chassis[0])
        
        # Generate waypoints using minimum step size.
        # Close balls get fewer waypoints; far balls are capped by configured maximum.
        min_waypoint_step_m = 0.015  # 2.5 cm spacing
        waypoints_from_distance = max(1, int(distance_to_ball / min_waypoint_step_m))
        num_waypoints = min(self.trajectory_waypoints, waypoints_from_distance)
        print(
            f"[DEBUG] Waypoints: step={min_waypoint_step_m*100:.1f}cm, "
            f"from_distance={waypoints_from_distance}, capped={num_waypoints}"
        )
        trajectory = []
        
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            wp_x = inlet_pos_chassis[0] + (ball_x - inlet_pos_chassis[0]) * alpha

            if close_mode:
                # Keep early segment near x-axis but allow bounded lateral drift by angle.
                # Then smoothly bend to the actual ball y-position.
                straight_portion = 0.35 + 0.45 * (1.0 - blend)  # 0.35 (far) -> 0.80 (very close)

                if alpha <= straight_portion:
                    # Lateral progress is capped by tan(max_angle).
                    if abs(dy) > 1e-6:
                        y_alpha_max = alpha * abs(dx) / abs(dy) * math.tan(max_deviation_angle)
                        y_alpha = min(alpha, y_alpha_max)
                    else:
                        y_alpha = 0.0
                else:
                    # Smooth transition from constrained segment to full trajectory.
                    beta = (alpha - straight_portion) / max(1e-6, (1.0 - straight_portion))
                    smooth = beta * beta * (3.0 - 2.0 * beta)  # smoothstep

                    if abs(dy) > 1e-6:
                        y_alpha_straight = straight_portion * abs(dx) / abs(dy) * math.tan(max_deviation_angle)
                        y_alpha_straight = min(straight_portion, y_alpha_straight)
                    else:
                        y_alpha_straight = 0.0

                    y_alpha = y_alpha_straight + (alpha - y_alpha_straight) * smooth

                wp_y = inlet_pos_chassis[1] + (ball_y - inlet_pos_chassis[1]) * y_alpha
            else:
                y_alpha = alpha
                wp_y = inlet_pos_chassis[1] + (ball_y - inlet_pos_chassis[1]) * y_alpha

            wp_theta = math.atan2(ball_y - inlet_pos_chassis[1], ball_x - inlet_pos_chassis[0])
            trajectory.append([wp_x, wp_y, wp_theta])
        
        print(f"[DEBUG] Created shaped trajectory with {len(trajectory)} waypoints")
        return trajectory
    
    def set_trajectory_planner(self, planner_type):
        """
        Change the trajectory planning method.
        
        Args:
            planner_type: One of TRAJECTORY_PLANNER_* constants
        """
        valid_planners = [TRAJECTORY_PLANNER_BEZIER, TRAJECTORY_PLANNER_SHAPED, 
                         TRAJECTORY_PLANNER_DIRECT, TRAJECTORY_PLANNER_ARC]
        if planner_type not in valid_planners:
            print(f"[WARNING] Invalid planner type '{planner_type}'. Valid options: {valid_planners}")
            return
        
        self.trajectory_planner = planner_type
        print(f"[INFO] Trajectory planner changed to: {planner_type}")
        
        # Clear existing trajectory to force replanning with new method
        self.trajectory = []
        self.current_traj_step = 0
    
    def compute_control(self):
        """
        Compute control commands to track current trajectory.
        Uses simple proportional control for first few trajectory steps.
        
        Returns:
            (linear_vel, angular_vel): Control commands in chassis frame
        """
        if not self.trajectory or self.current_traj_step >= len(self.trajectory):
            return 0.0, 0.0

        # Track only the first N waypoints in each planning cycle.
        max_track_steps = min(len(self.trajectory), max(1, self.lookahead_steps))
        if self.current_traj_step >= max_track_steps:
            return 0.0, 0.0
        
        # Get target waypoint
        target = self.trajectory[self.current_traj_step]
        target_x, target_y, target_theta = target
        
        # Current position (inlet frame in chassis coordinates)
        inlet_pos = self.T_chassis_to_inlet[:3, 3]
        current_x, current_y = inlet_pos[0], inlet_pos[1]
        
        # Error in position
        error_x = target_x - current_x
        error_y = target_y - current_y
        distance_error = math.hypot(error_x, error_y)
        
        # If close enough to current waypoint, advance to next
        if distance_error < self.waypoint_reach_threshold_m:
            self.current_traj_step += 1
            if self.current_traj_step >= max_track_steps:
                return 0.0, 0.0
            return self.compute_control()  # Recursive call for next waypoint
        
        # Desired heading to target waypoint
        desired_theta = math.atan2(error_y, error_x)
        
        # Angular error (assume chassis is aligned with inlet for now)
        # In practice, you'd need odometry or IMU for current heading
        inlet_rotation = self.T_chassis_to_inlet[:3, :3]
        current_theta = math.atan2(inlet_rotation[1, 0], inlet_rotation[0, 0])
        angular_error = desired_theta - current_theta
        
        # Normalize angle to [-pi, pi]
        angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))
        
        # Control commands
        linear_vel = self.k_linear * distance_error
        angular_vel = self.k_angular * angular_error
        
        # Apply minimum linear velocity to avoid slow approach
        if abs(linear_vel) > 1e-6:  # If non-zero, enforce minimum
            if linear_vel > 0:
                linear_vel = max(self.min_linear_vel, linear_vel)
            else:
                linear_vel = min(-self.min_linear_vel, linear_vel)
        
        # Apply minimum angular velocity to avoid slow turning
        if abs(angular_vel) > 1e-6:  # If non-zero, enforce minimum
            if angular_vel > 0:
                angular_vel = max(self.min_angular_vel, angular_vel)
            else:
                angular_vel = min(-self.min_angular_vel, angular_vel)
        
        # Clamp to chassis limits
        linear_vel = max(-self.chassis.max_linear_vel, 
                        min(self.chassis.max_linear_vel, linear_vel))
        max_angular_vel = math.radians(self.chassis.max_angular_vel_deg)
        angular_vel = max(-max_angular_vel, min(max_angular_vel, angular_vel))
        
        return linear_vel, angular_vel
    
    def update_ball_position(self, ball_pos_camera, force_replan=False):
        """Update ball position and replan trajectory if needed."""
        if ball_pos_camera is None:
            self.ball_detected = False
            self.ball_pos_ballfinder = None
            self.ball_pos_chassis = None
            return
        
        self.ball_detected = True
        self.ball_pos_ballfinder = np.array(ball_pos_camera, dtype=np.float64)
        self.ball_pos_chassis = self.ball_camera_to_chassis(ball_pos_camera)
        
        # Replan at fixed rate (for visibility) or when forced
        current_time = time.monotonic()
        if force_replan or (current_time - self.last_trajectory_update) >= self.trajectory_update_period:
            print(f"[DEBUG] Replanning trajectory (dt={current_time - self.last_trajectory_update:.2f}s)")
            self.trajectory = self.plan_trajectory(self.ball_pos_chassis)
            self.current_traj_step = 0
            self.last_trajectory_update = current_time
    
    def project_chassis_to_pixel(self, point_chassis, debug_label=""):
        """
        Project a 2D ground-level point in chassis frame to pixel coordinates.
        Simplified planar projection - ignores z-coordinate, treats all points as ground-level.
        """
        if point_chassis is None:
            return None
        
        # Extract x-y position from chassis frame (ignore z, treat as ground level)
        x_chassis = point_chassis[0]
        y_chassis = point_chassis[1]
        
        # For ground-level points, use BallFinder's projection approach:
        # Create a ray from camera through the ground point and find where it intersects
        # For simplicity, transform x-y to camera frame coordinates
        
        # Camera position in chassis frame
        camera_x = self.T_chassis_to_camera[0, 3]
        camera_y = self.T_chassis_to_camera[1, 3]
        
        # Point relative to camera position (still in chassis x-y plane)
        dx = x_chassis - camera_x
        dy = y_chassis - camera_y
        
        # Convert to robot frame at camera origin (x=forward, y=left)
        # Camera convention: x=right, y=down, z=forward
        # BallFinder inverse: p_cam[0] = -robot_y, p_cam[1] = -robot_x
        p_cam_x = -dy   # Camera x (right) = robot y (left)
        p_cam_y = -dx   # Camera y (down) = robot x (forward)
        
        # For 90° downward camera at height H looking at ground (z=0):
        # Distance to ground point determines cam_z via ray-plane intersection
        # Simplified: cam_z ≈ camera_height (all ground points are "in front" at distance H)
        camera_height = 0.26
        p_cam_z = camera_height
        
        if debug_label:
            print(f"[DEBUG] {debug_label} chassis_xy=({x_chassis:.3f}, {y_chassis:.3f}), " 
                  f"rel_to_cam=({dx:.3f}, {dy:.3f}), p_cam=({p_cam_x:.3f}, {p_cam_y:.3f}, {p_cam_z:.3f})")
        
        # Project to pixel using BallFinder's camera matrix
        K = self.ball_finder.K
        x_proj = K[0, 0] * p_cam_x / p_cam_z + K[0, 2]
        y_proj = K[1, 1] * p_cam_y / p_cam_z + K[1, 2]
        
        # Check if in frame bounds
        in_bounds = (0 <= x_proj < self.ball_finder.frame_w and 0 <= y_proj < self.ball_finder.frame_h)
        if debug_label and not in_bounds:
            print(f"[DEBUG] {debug_label} out of bounds: pixel=({x_proj:.1f}, {y_proj:.1f})")
        
        # Return pixel even if slightly out of bounds (for drawing)
        return (int(x_proj), int(y_proj))
    
    def draw_trajectory_on_frame(self, frame, ball_pixel_pos=None):
        """
        Draw trajectory information on camera frame.
        
        Args:
            frame: OpenCV BGR image
            ball_pixel_pos: Ball pixel position (u, v) or None
        """
        text_y = 60
        line_height = 25

        # Coordinate debug block
        cv2.putText(
            frame,
            f"BallFinder center: {fmt_cm(self.ball_pos_ballfinder) if self.ball_pos_ballfinder is not None else 'None'}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Ball center (chassis): {fmt_cm(self.ball_pos_chassis) if self.ball_pos_chassis is not None else 'None'}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )
        inlet_pos_chassis = self.T_chassis_to_inlet[:3, 3]
        cv2.putText(
            frame,
            f"Inlet center (chassis): {fmt_cm(inlet_pos_chassis)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
        )
        text_y = 90
        
        # Ball detection status
        if self.ball_detected and self.ball_pos_chassis is not None:
            status_text = f"Ball detected: {fmt_cm(self.ball_pos_chassis)} (chassis)"
            cv2.putText(frame, status_text, (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw circle at ball pixel position
            if ball_pixel_pos is not None:
                cv2.circle(frame, (int(ball_pixel_pos[0]), int(ball_pixel_pos[1])), 
                          8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Ball: NOT DETECTED", (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        text_y += line_height
        
        # Draw trajectory on frame
        if self.trajectory:
            print(f"[DEBUG] Drawing trajectory with {len(self.trajectory)} waypoints")
            # Project inlet position to pixel
            inlet_pos_3d = self.T_chassis_to_inlet[:3, 3]
            inlet_pixel_raw = self.project_chassis_to_pixel(inlet_pos_3d, "inlet")
            
            # Always draw inlet by clamping to frame edges if needed
            inlet_pixel = None
            if inlet_pixel_raw is not None:
                # Clamp to frame bounds so inlet is always visible
                inlet_x = max(5, min(self.ball_finder.frame_w - 5, inlet_pixel_raw[0]))
                inlet_y = max(5, min(self.ball_finder.frame_h - 5, inlet_pixel_raw[1]))
                inlet_pixel = (inlet_x, inlet_y)
                cv2.circle(frame, inlet_pixel, 8, (0, 0, 255), -1)  # RED filled circle
                cv2.putText(frame, "INLET", (inlet_pixel[0] + 10, inlet_pixel[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"[DEBUG] Inlet drawn at pixel {inlet_pixel} (raw: {inlet_pixel_raw})")
            else:
                print(f"[DEBUG] Inlet behind camera, cannot project")
            
            # Draw trajectory waypoints
            prev_pixel = inlet_pixel
            visible_count = 0
            for i, waypoint in enumerate(self.trajectory):
                # Waypoint is [x, y, theta], construct 3D point at ground level
                wp_3d = np.array([waypoint[0], waypoint[1], 0.0])
                wp_pixel_raw = self.project_chassis_to_pixel(wp_3d, f"wp{i}")
                
                if wp_pixel_raw is not None:
                    visible_count += 1
                    # Clamp waypoint to frame bounds (like inlet) so trajectory stays visible
                    wp_x = max(0, min(self.ball_finder.frame_w - 1, wp_pixel_raw[0]))
                    wp_y = max(0, min(self.ball_finder.frame_h - 1, wp_pixel_raw[1]))
                    wp_pixel = (wp_x, wp_y)
                    
                    # Draw line from previous point
                    if prev_pixel is not None:
                        # Color gradient: cyan -> yellow
                        color_ratio = i / len(self.trajectory)
                        color = (int(255 * (1 - color_ratio)), 255, int(255 * color_ratio))
                        cv2.line(frame, prev_pixel, wp_pixel, color, 3)  # Thicker line
                    
                    # Draw waypoint marker
                    if i == self.current_traj_step:
                        # Current target waypoint - larger green circle with label
                        cv2.circle(frame, wp_pixel, 7, (0, 255, 0), -1)
                        cv2.putText(frame, "NEXT", (wp_pixel[0] + 10, wp_pixel[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Other waypoints - small circles
                        cv2.circle(frame, wp_pixel, 4, (255, 255, 0), -1)
                    
                    prev_pixel = wp_pixel
            
            print(f"[DEBUG] Drew {visible_count}/{len(self.trajectory)} visible waypoints")
        else:
            print(f"[DEBUG] No trajectory to draw")
            
            # Trajectory info text
            traj_text = f"Trajectory: {len(self.trajectory)} waypoints, step {self.current_traj_step}"
            cv2.putText(frame, traj_text, (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            text_y += line_height
            
            # Current target waypoint
            if self.current_traj_step < len(self.trajectory):
                wp = self.trajectory[self.current_traj_step]
                wp_text = f"Target WP: x={wp[0]*100:.1f}cm y={wp[1]*100:.1f}cm theta={math.degrees(wp[2]):.1f}deg"
                cv2.putText(frame, wp_text, (10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "Trajectory: NONE", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame


class LongRangeSearchApproachController:
    """Owns long-range search/approach state machine and command generation."""

    def __init__(self, search_scan_angle_deg=360.0, far_detector_ready=False):
        self.search_scan_enabled = True
        self.search_scan_angle_deg = max(1.0, float(search_scan_angle_deg))
        self.search_scan_speed_deg_s = 40.0
        self.search_scan_far_speed_deg_s = 20.0
        self.search_scan_far_detect_slowdown_factor = 0.2
        self.search_scan_far_detect_slowdown_hold_s = 0.8
        self.search_scan_far_detect_slowdown_until = 0.0
        self.search_scan_far_detect_confirm_frames = 1
        self.search_scan_far_detect_streak = 0
        self.search_scan_direction = 1.0
        self.search_scan_active = False
        self.search_scan_start_time = 0.0
        self.search_scan_duration_s = self.search_scan_angle_deg / self.search_scan_speed_deg_s
        self.search_scan_max_sweeps = 2
        self.search_scan_far_max_sweeps = 2
        self.search_scan_completed_sweeps = 0
        self.search_scan_exhausted = False
        self.search_scan_phase = "normal"  # normal -> wait_far -> far -> far_wait_cycle -> far ... or far_approach
        self.search_scan_far_wait_s = 1.0
        self.search_scan_far_wait_until = 0.0
        self.search_scan_far_cycle_wait_s = 5.0
        self.search_scan_far_cycle_wait_until = 0.0
        self.search_scan_far_cycle_count = 0
        self.search_scan_far_max_cycles = 3
        self.far_recover_forward_s = 1.0
        self.far_recover_forward_linear_vel = 0.05
        self.far_recover_forward_until = 0.0

        self.far_detector_ready = bool(far_detector_ready)
        self.far_target_last_bearing_deg = None
        self.far_target_last_direction_text = ""
        self.far_approach_linear_vel = 0.06
        self.far_approach_kp_angular = 2.2  # deg/s per deg bearing error
        self.far_approach_max_angular_deg_s = 35.0
        self.far_approach_center_tolerance_deg = 4.0
        self.far_approach_lost_spin_deg_s = 0.0
        self.far_target_lock_active = False
        self.far_target_lock_px = None
        self.far_target_lock_bearing_deg = None
        self.far_target_lock_conf = 0.0
        self.far_target_lost_frames = 0
        self.far_target_match_max_px = 80.0
        self.far_target_max_lost_frames = 10
        self.far_target_retarget_period_s = 0.5
        self.far_target_retarget_next_time = 0.0
        self.far_target_retarget_xproj_hysteresis_m = 0.12

    def _get_current_sweep_speed_deg_s(self, now=None):
        if self.search_scan_phase == "far":
            speed = self.search_scan_far_speed_deg_s
            if now is None:
                now = time.monotonic()
            if now < self.search_scan_far_detect_slowdown_until:
                speed *= self.search_scan_far_detect_slowdown_factor
            return speed
        return self.search_scan_speed_deg_s

    def _get_current_sweep_duration_s(self, now=None):
        current_speed = self._get_current_sweep_speed_deg_s(now=now)
        return self.search_scan_angle_deg / max(1e-6, current_speed)

    def set_far_detector_ready(self, ready):
        self.far_detector_ready = bool(ready)

    def clear_target_lock(self):
        self.far_target_lock_active = False
        self.far_target_lock_px = None
        self.far_target_lock_bearing_deg = None
        self.far_target_lock_conf = 0.0
        self.far_target_lost_frames = 0
        self.far_target_retarget_next_time = 0.0

    def reset_search_cycle(self):
        self.search_scan_active = False
        self.search_scan_completed_sweeps = 0
        self.search_scan_exhausted = False
        self.search_scan_phase = "normal"
        self.search_scan_far_cycle_count = 0
        self.search_scan_far_detect_slowdown_until = 0.0
        self.search_scan_far_detect_streak = 0
        self.far_recover_forward_until = 0.0

    def on_autonomy_toggled(self, running):
        if not running:
            self.search_scan_active = False
            self.clear_target_lock()
        self.search_scan_completed_sweeps = 0
        self.search_scan_exhausted = False
        self.search_scan_phase = "normal"
        self.search_scan_far_detect_slowdown_until = 0.0
        self.search_scan_far_detect_streak = 0
        self.far_recover_forward_until = 0.0

    def toggle_search_scan(self):
        self.search_scan_enabled = not self.search_scan_enabled
        self.search_scan_active = False
        self.clear_target_lock()
        self.search_scan_completed_sweeps = 0
        self.search_scan_exhausted = False
        self.search_scan_phase = "normal"
        self.search_scan_far_detect_slowdown_until = 0.0
        self.search_scan_far_detect_streak = 0
        self.far_recover_forward_until = 0.0
        return self.search_scan_enabled

    def on_inlet_ball_detected(self):
        if self.search_scan_phase in ("wait_far", "far", "far_approach") or self.search_scan_active:
            print("[INFO] Ball detected in inlet camera: switching to inlet tracking")
        self.clear_target_lock()
        self.reset_search_cycle()

    def compute_commands(
        self,
        running,
        ball_detected,
        user_inlet_enabled,
        planned_linear_vel,
        planned_angular_vel,
        far_ball_detected,
        far_ball_center_px,
        far_ball_bearing_deg,
        far_ball_best_conf,
        far_ball_direction_text,
        chassis_max_linear_vel,
    ):
        """
        Compute autonomous commands with long-range search/approach state machine.

        Returns:
            (linear_vel_cmd, angular_vel_cmd, inlet_active, execute_motion)
        """
        if running and ball_detected:
            self.on_inlet_ball_detected()
            return planned_linear_vel, planned_angular_vel, (user_inlet_enabled and ball_detected), True

        if not (running and self.search_scan_enabled):
            self.reset_search_cycle()
            return 0.0, 0.0, False, False

        now = time.monotonic()

        # Immediate handoff: if inlet camera loses ball but long-range camera sees one,
        # switch straight to FAR approach using the long-range target.
        if (not ball_detected) and far_ball_detected:
            if self.search_scan_phase != "far_approach":
                print(
                    f"[INFO] Inlet lost ball, FAR target available: switching to FAR approach "
                    f"(bearing={far_ball_bearing_deg:+.1f}deg {far_ball_direction_text})"
                )
            self.search_scan_phase = "far_approach"
            self.search_scan_active = False
            self.search_scan_exhausted = False
            self.search_scan_completed_sweeps = 0
            self.search_scan_far_cycle_count = 0
            self.far_target_lock_active = True
            self.far_target_lock_px = far_ball_center_px
            self.far_target_lock_bearing_deg = far_ball_bearing_deg
            self.far_target_lock_conf = far_ball_best_conf
            self.far_target_lost_frames = 0
            if self.far_target_retarget_next_time <= 0.0:
                self.far_target_retarget_next_time = now + self.far_target_retarget_period_s

        if self.search_scan_phase == "done":
            self.search_scan_exhausted = True
            self.clear_target_lock()
            return 0.0, 0.0, False, False

        if self.search_scan_phase == "far_recover_forward":
            if now < self.far_recover_forward_until:
                linear_vel_cmd = min(self.far_recover_forward_linear_vel, chassis_max_linear_vel)
                return linear_vel_cmd, 0.0, False, True

            # Recovery complete: resume FAR search sweeping
            self.search_scan_phase = "far"
            self.search_scan_active = False
            self.search_scan_exhausted = False
            self.search_scan_completed_sweeps = 0
            self.search_scan_far_detect_slowdown_until = 0.0
            self.search_scan_far_detect_streak = 0
            self.clear_target_lock()
            print("[INFO] FAR recovery forward complete: resuming FAR search")
            return 0.0, 0.0, False, False

        if self.search_scan_phase == "far_approach":
            if not far_ball_detected:
                self.search_scan_phase = "far_recover_forward"
                self.far_recover_forward_until = now + self.far_recover_forward_s
                self.clear_target_lock()
                print(
                    f"[INFO] FAR target lost: moving forward for {self.far_recover_forward_s:.1f}s, then resuming FAR search"
                )
                linear_vel_cmd = min(self.far_recover_forward_linear_vel, chassis_max_linear_vel)
                return linear_vel_cmd, 0.0, False, True

            bearing_for_control = far_ball_bearing_deg
            if bearing_for_control is not None:
                angular_cmd_deg_s = FAR_APPROACH_TURN_SIGN * self.far_approach_kp_angular * bearing_for_control
                angular_cmd_deg_s = max(
                    -self.far_approach_max_angular_deg_s,
                    min(self.far_approach_max_angular_deg_s, angular_cmd_deg_s),
                )
                if abs(bearing_for_control) <= self.far_approach_center_tolerance_deg:
                    linear_vel_cmd = min(self.far_approach_linear_vel, chassis_max_linear_vel)
                else:
                    # Keep approaching (reduced speed) instead of spin-in-place.
                    linear_vel_cmd = min(0.5 * self.far_approach_linear_vel, chassis_max_linear_vel)
                angular_vel_cmd = math.radians(angular_cmd_deg_s)
            else:
                linear_vel_cmd = 0.0
                angular_vel_cmd = 0.0

            return linear_vel_cmd, angular_vel_cmd, False, True

        if self.search_scan_phase == "far_wait_cycle":
            if now >= self.search_scan_far_cycle_wait_until:
                self.search_scan_phase = "far"
                self.search_scan_active = False
                self.search_scan_completed_sweeps = 0
                self.search_scan_exhausted = False
                self.search_scan_far_detect_slowdown_until = 0.0
                self.search_scan_far_detect_streak = 0
                self.far_recover_forward_until = 0.0
                print(f"[INFO] FAR cycle restart: running {self.search_scan_far_max_sweeps} sweeps")
            return 0.0, 0.0, False, False

        if self.search_scan_phase == "wait_far":
            if now >= self.search_scan_far_wait_until:
                if self.far_detector_ready:
                    self.search_scan_phase = "far"
                    self.search_scan_active = False
                    self.search_scan_completed_sweeps = 0
                    self.search_scan_exhausted = False
                    self.search_scan_direction = 1.0
                    self.search_scan_far_detect_slowdown_until = 0.0
                    self.search_scan_far_detect_streak = 0
                    self.far_recover_forward_until = 0.0
                    print("[INFO] Starting FAR search phase: 2 sweeps with CSI0 NCNN detection")
                else:
                    self.search_scan_phase = "done"
                    self.search_scan_exhausted = True
                    print("[INFO] Far search skipped (CSI0 NCNN unavailable)")
            return 0.0, 0.0, False, False

        phase_name = "FAR" if self.search_scan_phase == "far" else "NORMAL"
        phase_max_sweeps = self.search_scan_far_max_sweeps if self.search_scan_phase == "far" else self.search_scan_max_sweeps

        if self.search_scan_phase == "far":
            if far_ball_detected:
                if now >= self.search_scan_far_detect_slowdown_until:
                    print(
                        f"[INFO] FAR detection seen: slowing sweep to "
                        f"{self.search_scan_far_detect_slowdown_factor:.2f}x"
                    )
                self.search_scan_far_detect_slowdown_until = max(
                    self.search_scan_far_detect_slowdown_until,
                    now + self.search_scan_far_detect_slowdown_hold_s,
                )

                self.search_scan_far_detect_streak += 1
                if self.search_scan_far_detect_streak >= self.search_scan_far_detect_confirm_frames:
                    self.search_scan_active = False
                    self.search_scan_exhausted = False
                    self.search_scan_phase = "far_approach"
                    self.search_scan_far_cycle_count = 0
                    self.far_target_lock_active = True
                    self.far_target_lock_px = far_ball_center_px
                    self.far_target_lock_bearing_deg = far_ball_bearing_deg
                    self.far_target_lock_conf = far_ball_best_conf
                    self.far_target_lost_frames = 0
                    print(
                        f"[INFO] FAR target acquired: switching to FAR approach. "
                        f"Direction={far_ball_direction_text}, bearing={far_ball_bearing_deg:+.1f}deg"
                    )
                    return 0.0, 0.0, False, True
            else:
                self.search_scan_far_detect_streak = 0

        if self.search_scan_exhausted:
            return 0.0, 0.0, False, False

        if not self.search_scan_active:
            self.search_scan_active = True
            self.search_scan_start_time = now
            current_speed = self._get_current_sweep_speed_deg_s(now=now)
            print(
                f"[INFO] {phase_name} search sweep start: angle={self.search_scan_angle_deg:.1f}deg, "
                f"speed={current_speed:.1f}deg/s, "
                f"direction={'CCW' if self.search_scan_direction > 0 else 'CW'}"
            )

        sweep_elapsed = now - self.search_scan_start_time
        if sweep_elapsed >= self._get_current_sweep_duration_s(now=now):
            self.search_scan_completed_sweeps += 1
            if self.search_scan_completed_sweeps >= phase_max_sweeps:
                self.search_scan_active = False
                self.search_scan_exhausted = True
                if self.search_scan_phase == "normal":
                    self.search_scan_phase = "wait_far"
                    self.search_scan_far_wait_until = now + self.search_scan_far_wait_s
                    print(
                        f"[INFO] Normal search stopped after {self.search_scan_completed_sweeps} sweeps; "
                        f"waiting {self.search_scan_far_wait_s:.1f}s before FAR search"
                    )
                elif self.search_scan_phase == "far":
                    self.search_scan_far_cycle_count += 1
                    if self.search_scan_far_cycle_count >= self.search_scan_far_max_cycles:
                        self.search_scan_phase = "done"
                        self.search_scan_far_detect_slowdown_until = 0.0
                        self.search_scan_far_detect_streak = 0
                        self.far_recover_forward_until = 0.0
                        print(
                            f"[INFO] FAR search stopped after {self.search_scan_far_cycle_count} cycles with no detections"
                        )
                    else:
                        self.search_scan_phase = "far_wait_cycle"
                        self.search_scan_exhausted = False
                        self.search_scan_far_cycle_wait_until = now + self.search_scan_far_cycle_wait_s
                        self.search_scan_far_detect_slowdown_until = 0.0
                        self.search_scan_far_detect_streak = 0
                        self.far_recover_forward_until = 0.0
                        print(
                            f"[INFO] FAR cycle complete ({self.search_scan_completed_sweeps} sweeps); "
                            f"waiting {self.search_scan_far_cycle_wait_s:.1f}s "
                            f"(cycle {self.search_scan_far_cycle_count}/{self.search_scan_far_max_cycles})"
                        )
                else:
                    self.search_scan_phase = "done"
                    self.search_scan_far_detect_slowdown_until = 0.0
                    self.search_scan_far_detect_streak = 0
                    self.far_recover_forward_until = 0.0
                    print(
                        f"[INFO] Far search stopped: no ball detected after {self.search_scan_completed_sweeps} sweeps"
                    )
                return 0.0, 0.0, False, False

            self.search_scan_direction *= -1.0
            self.search_scan_start_time = now
            print(
                f"[INFO] {phase_name} search sweep reverse: direction={'CCW' if self.search_scan_direction > 0 else 'CW'} "
                f"({self.search_scan_completed_sweeps}/{phase_max_sweeps})"
            )

        if not self.search_scan_exhausted:
            current_speed = self._get_current_sweep_speed_deg_s(now=now)
            return 0.0, math.radians(current_speed * self.search_scan_direction), False, True

        return 0.0, 0.0, False, False


def main():
    parser = argparse.ArgumentParser(
        description="Ball collection with visual servoing control"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        choices=[0, 1],
        help="Camera source: 0 (CSI 0) or 1 (CSI 1) (default: 1)"
    )
    parser.add_argument(
        "--search-scan-angle-deg",
        type=float,
        default=360.0,
        help="Search mode sweep angle when no ball is detected (default: 360 deg)",
    )
    args = parser.parse_args()
    
    # Initialize servo communication
    portHandler = PortHandler('/dev/ttyACM0')
    packetHandler = sms_sts(portHandler)
    
    if not portHandler.openPort():
        print("Failed to open servo port")
        return
    
    if not portHandler.setBaudRate(1000000):
        print("Failed to set baudrate")
        portHandler.closePort()
        return
    
    print("Servo communication initialized")
    
    # Initialize chassis
    chassis = Chassis(
        packetHandler,
        shaft_distance=0.2,
        wheel_diameter=0.1,
        inlet_disk_diameter=0.05,
        wheel_reduction=1,
        inlet_reduction=0.1,
        max_linear_vel=0.225,  # 1.5x faster for visual servoing
        max_angular_vel_deg=60,  # 1.5x faster angular velocity
        max_inlet_vel=4.5
    )
    
    # Transformation matrices (ADJUST THESE FOR YOUR ROBOT)
    # Example: Camera mounted 10cm forward, 0cm lateral, 26cm up from chassis center
    T_chassis_to_camera = np.array([
        [1, 0, 0, 0.0866],  # Camera 10cm forward
        [0, 1, 0, 0.00],  # Camera at center laterally
        [0, 0, 1, 0.306],  # Camera 30.6cm above ground
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    # Inlet in front of camera (so it's visible in camera view)
    T_chassis_to_inlet = np.array([
        [1, 0, 0, -0.032],  # Inlet 5cm forward (5cm behind of camera)
        [0, 1, 0, 0.00],  # Inlet at center
        [0, 0, 1, 0.00],  # Inlet at ground level
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    # Load camera calibration
    K, dist = load_calibration(CALIB_FILE)
    print("Camera calibration loaded")
    
    # Initialize IMX500
    imx500 = IMX500(MODEL_PATH)
    intr = imx500.network_intrinsics
    if not intr:
        intr = NetworkIntrinsics()
        intr.task = "object detection"
    if intr.task != "object detection":
        raise RuntimeError("Loaded network is not object detection")
    
    intr.update_with_defaults()
    labels = intr.labels if intr.labels is not None else default_coco_labels()
    
    # Initialize camera
    picam2 = Picamera2(args.camera)
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
        controls={"FrameRate": float(intr.inference_rate)},
        buffer_count=8
    )
    picam2.configure(config)
    imx500.show_network_fw_progress_bar()
    picam2.start()
    time.sleep(0.5)
    
    print("Camera initialized")

    # Initialize optional long-range camera preview (always CSI0 when primary is not CSI0)
    picam2_csi0 = None
    csi0_preview_error_reported = False
    far_detector_model = None
    far_detector_target_cls = None
    far_detector_ready = False
    if args.camera != 0:
        try:
            picam2_csi0 = Picamera2(0)
            config_csi0 = picam2_csi0.create_preview_configuration(
                main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
                buffer_count=4,
            )
            picam2_csi0.configure(config_csi0)
            picam2_csi0.start()
            time.sleep(0.2)
            print("Long-range camera preview initialized on CSI0")
        except Exception as exc:
            print(f"[WARNING] Could not initialize CSI0 long-range preview: {exc}")
            picam2_csi0 = None

    if picam2_csi0 is not None:
        if YOLO is None:
            print("[WARNING] Ultralytics not available: far NCNN detection disabled")
        else:
            try:
                model_path = Path(FAR_NCNN_MODEL_PATH)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model path not found: {model_path}")
                far_detector_model = YOLO(str(model_path))
                far_detector_target_cls = find_class_id(far_detector_model, "sports ball")
                far_detector_ready = True
                print(
                    f"Far NCNN detector initialized on CSI0: model={model_path}, imgsz={FAR_NCNN_IMG_SIZE}"
                )
            except Exception as exc:
                print(f"[WARNING] Could not initialize far NCNN detector: {exc}")
    
    # Initialize BallFinder
    ball_finder = BallFinder(
        camera_matrix=K,
        dist_coeffs=dist,
        frame_w=FRAME_W,
        frame_h=FRAME_H,
        camera_height_m=0.26,
        camera_pitch_deg=90.0,
        ball_radius_m=0.0335,
        ema_alpha=0.25,
        use_center_gate=False,
    )
    
    print("BallFinder initialized")
    
    # Initialize visual controller
    controller = InletVisualController(
        chassis=chassis,
        ball_finder=ball_finder,
        T_chassis_to_camera=T_chassis_to_camera,
        T_chassis_to_inlet=T_chassis_to_inlet,
        control_rate_hz=10.0,
        trajectory_waypoints=20,
        lookahead_steps=1,
        target_approach_distance=0.010,
    )
    
    print("Visual controller initialized")
    print(f"Trajectory planner: {controller.trajectory_planner}")
    print("=== CONTROL MODES ===")
    print("Controls:")
    print("  SPACE = start/stop autonomous planning")
    print("  'm' = toggle manual/auto control mode")
    print("  'i' = toggle inlet on/off")
    print("  's' = toggle search scan mode (auto sweep when no ball)")
    print("  'r' = reset collected ball counter")
    print("  'q' = quit")
    print("  1-4 = select trajectory planner")
    print("    1: Bézier curve (default)")
    print("    2: Shaped approach")
    print("    3: Direct line")
    print("    4: Arc")
    print("Manual keyboard control (when in manual mode):")
    print("  UP/DOWN arrows = linear velocity +/- 0.01 m/s")
    print("  LEFT/RIGHT arrows = angular velocity +/- 1 deg/s")
    if picam2_csi0 is not None:
        print("Dual-view display enabled: main + long-range camera (CSI0)")
    print("Starting control loop...")

    chassis.stop_event.clear()
    chassis.run_event.clear()
    motion_thread = threading.Thread(target=chassis.motionControlWorker, daemon=True)
    motion_thread.start()
    
    # Control loop state
    running = False
    user_inlet_enabled = True  # User's inlet preference
    manual_mode = False  # Manual keyboard control vs autonomous

    # Optional long-range search/approach controller
    long_range_controller = LongRangeSearchApproachController(
        search_scan_angle_deg=args.search_scan_angle_deg,
        far_detector_ready=far_detector_ready,
    )
    
    # Manual control state
    manual_linear_vel = 0.0
    manual_angular_vel = 0.0
    key_linear_step = 0.01  # m/s increment per keypress (matching keyboad_input.py)
    key_angular_step = 1.0  # deg/s increment per keypress (matching keyboad_input.py)

    # Post-collection behavior: keep all motors on briefly after ball disappears
    post_collect_duration_s = 1.0
    post_collect_until = 0.0
    post_collect_cooldown_s = 3.0  # Cooldown period to prevent repeated triggering
    post_collect_cooldown_until = 0.0
    prev_ball_detected = False
    ball_detected_since = 0.0  # Timestamp when ball first detected
    min_detection_duration_s = 1.5  # Minimum time ball must be detected to be considered real
    collect_near_inlet_distance_m = 0.07  # Count as collected if ball center gets this close to inlet
    episode_min_ball_inlet_distance_m = float('inf')
    last_motion_linear_cmd = 0.0
    last_motion_angular_cmd = 0.0
    balls_collected_count = 0
    
    try:
        while True:
            loop_start = time.monotonic()
            
            # Capture frame and detect ball
            request = picam2.capture_request()
            md = request.get_metadata()
            
            with MappedArray(request, "main") as m:
                frame_rgb = m.array.copy()
            
            # Detect ball
            dets = []
            np_outputs = imx500.get_outputs(md, add_batch=True)
            if np_outputs is not None:
                input_w, input_h = imx500.get_input_size()
                
                if intr.postprocess == "nanodet":
                    (boxes, scores, classes) = postprocess_nanodet_detection(
                        outputs=np_outputs[0], conf=CONF_MIN, iou_thres=0.65, max_out_dets=10
                    )[0]
                else:
                    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
                    if intr.bbox_normalization:
                        boxes = boxes / input_h
                    if intr.bbox_order == "xy":
                        boxes = boxes[:, [1, 0, 3, 2]]
                
                for box, score, cat in zip(boxes, scores, classes):
                    conf = float(score)
                    if conf < CONF_MIN:
                        continue
                    
                    cat_i = int(cat)
                    label = labels[cat_i] if 0 <= cat_i < len(labels) else str(cat_i)
                    if label not in TARGET_LABELS:
                        continue
                    
                    x, y, w, h = imx500.convert_inference_coords(box, md, picam2)
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    if w <= 0 or h <= 0:
                        continue
                    
                    u, v = ball_finder.compute_ball_center(x, y, w, h)
                    score_total, d_center = ball_finder.detection_score(conf, u, v)
                    if score_total is None:
                        continue
                    dets.append((score_total, conf, label, x, y, w, h, u, v))
            
            request.release()
            
            # Process best detection
            ball_pixel_pos = None
            ball_pos_camera = None
            
            if dets:
                chosen = max(dets, key=lambda t: t[0])
                score_total, conf, label, x, y, w, h, u, v = chosen
                ball_pixel_pos = (u, v)
                
                # Get 3D position
                p_cam, _ = ball_finder.project_to_3d(u, v)
                ball_pos_camera = ball_finder.camera_to_robot_frame(p_cam)
                
                # Update controller
                controller.update_ball_position(ball_pos_camera)
                
                # Set inlet based on user preference AND ball detection
                with chassis.state_lock:
                    chassis.inlet_enabled = user_inlet_enabled  # Use user preference
            else:
                controller.update_ball_position(None)
                with chassis.state_lock:
                    chassis.inlet_enabled = False  # No ball, turn off inlet

            # Capture CSI0 frame once per loop (for preview + optional far NCNN detection)
            frame_csi0_rgb = None
            frame_csi0_bgr = None
            far_ball_detected = False
            far_ball_best_conf = 0.0
            far_ball_center_px = None
            far_ball_bearing_deg = None
            far_ball_direction_text = ""
            if picam2_csi0 is not None:
                try:
                    frame_csi0_rgb = picam2_csi0.capture_array("main")
                    frame_csi0_bgr = cv2.cvtColor(frame_csi0_rgb, cv2.COLOR_RGB2BGR)
                    if frame_csi0_bgr.shape[0] != FRAME_H or frame_csi0_bgr.shape[1] != FRAME_W:
                        frame_csi0_bgr = cv2.resize(frame_csi0_bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

                    if (
                        far_detector_ready
                        and running
                        and long_range_controller.search_scan_enabled
                        and frame_csi0_rgb is not None
                    ):
                        det = far_detector_model.predict(
                            source=frame_csi0_rgb,
                            imgsz=FAR_NCNN_IMG_SIZE,
                            conf=FAR_NCNN_CONF,
                            max_det=10,
                            verbose=False,
                            device="cpu",
                        )[0]

                        far_candidates = []
                        if det.boxes is not None and len(det.boxes) > 0:
                            fx_far_px = (FRAME_W * 0.5) / math.tan(math.radians(FAR_CSI0_HFOV_DEG * 0.5))
                            for i in range(len(det.boxes)):
                                cls = int(det.boxes.cls[i].item())
                                if cls != far_detector_target_cls:
                                    continue
                                conf = float(det.boxes.conf[i].item())
                                x0, y0, x1, y1 = [int(v) for v in det.boxes.xyxy[i].tolist()]
                                center_x = int((x0 + x1) * 0.5)
                                center_y = int((y0 + y1) * 0.5)

                                box_w = max(1.0, float(x1 - x0))
                                box_h = max(1.0, float(y1 - y0))
                                size_px = max(box_w, box_h)

                                x_offset_px = float(center_x) - (FRAME_W * 0.5)
                                norm_offset = x_offset_px / max(1.0, FRAME_W * 0.5)
                                norm_offset = max(-1.0, min(1.0, norm_offset))
                                bearing_deg = norm_offset * (FAR_CSI0_HFOV_DEG * 0.5)
                                bearing_rad = math.radians(bearing_deg)

                                # Approximate forward distance from bbox size, then project on chassis x-axis.
                                range_m = (fx_far_px * FAR_BALL_DIAMETER_M) / size_px
                                x_proj_m = max(0.0, range_m * math.cos(bearing_rad))

                                far_candidates.append({
                                    "conf": conf,
                                    "x0": x0,
                                    "y0": y0,
                                    "x1": x1,
                                    "y1": y1,
                                    "center": (center_x, center_y),
                                    "bearing_deg": bearing_deg,
                                    "x_proj_m": x_proj_m,
                                })

                        selected_candidate = None
                        if far_candidates:
                            if (
                                long_range_controller.search_scan_phase == "far_approach"
                                and long_range_controller.far_target_lock_active
                                and long_range_controller.far_target_lock_px is not None
                            ):
                                now_sel = time.monotonic()

                                # Low-frequency retarget: periodically switch to newly closest ball.
                                if now_sel >= long_range_controller.far_target_retarget_next_time:
                                    new_closest = min(
                                        far_candidates,
                                        key=lambda c: (c["x_proj_m"], -c["conf"]),
                                    )

                                    # Hysteresis: if current lock is almost as close, keep it to avoid oscillation.
                                    current_lock_candidate = None
                                    best_lock_dist = float("inf")
                                    for cand in far_candidates:
                                        cx, cy = cand["center"]
                                        dx = float(cx - long_range_controller.far_target_lock_px[0])
                                        dy = float(cy - long_range_controller.far_target_lock_px[1])
                                        d = math.hypot(dx, dy)
                                        if d < best_lock_dist:
                                            best_lock_dist = d
                                            current_lock_candidate = cand

                                    if (
                                        current_lock_candidate is not None
                                        and best_lock_dist <= long_range_controller.far_target_match_max_px
                                    ):
                                        delta_xproj = (
                                            current_lock_candidate["x_proj_m"] - new_closest["x_proj_m"]
                                        )
                                        if delta_xproj <= long_range_controller.far_target_retarget_xproj_hysteresis_m:
                                            selected_candidate = current_lock_candidate
                                        else:
                                            selected_candidate = new_closest
                                    else:
                                        selected_candidate = new_closest

                                    long_range_controller.far_target_retarget_next_time = (
                                        now_sel + long_range_controller.far_target_retarget_period_s
                                    )
                                    long_range_controller.far_target_lost_frames = 0
                                else:
                                    # Between retarget ticks, keep matching current lock for stability.
                                    best_match = None
                                    best_dist = float("inf")
                                    for cand in far_candidates:
                                        cx, cy = cand["center"]
                                        dx = float(cx - long_range_controller.far_target_lock_px[0])
                                        dy = float(cy - long_range_controller.far_target_lock_px[1])
                                        d = math.hypot(dx, dy)
                                        if d < best_dist:
                                            best_dist = d
                                            best_match = cand

                                    if best_match is not None and best_dist <= long_range_controller.far_target_match_max_px:
                                        selected_candidate = best_match
                                        long_range_controller.far_target_lost_frames = 0
                                    else:
                                        long_range_controller.far_target_lost_frames += 1
                                        if long_range_controller.far_target_lost_frames > long_range_controller.far_target_max_lost_frames:
                                            print(
                                                f"[INFO] FAR target lock lost after {long_range_controller.far_target_lost_frames} frames; allowing reacquire"
                                            )
                                            long_range_controller.clear_target_lock()

                            if selected_candidate is None and (
                                long_range_controller.search_scan_phase != "far_approach" or not long_range_controller.far_target_lock_active
                            ):
                                if long_range_controller.search_scan_phase == "far":
                                    selected_candidate = min(
                                        far_candidates,
                                        key=lambda c: (c["x_proj_m"], -c["conf"]),
                                    )
                                else:
                                    selected_candidate = max(far_candidates, key=lambda c: c["conf"])
                                long_range_controller.far_target_lost_frames = 0

                        if selected_candidate is not None:
                            far_ball_detected = True
                            far_ball_best_conf = float(selected_candidate["conf"])
                            far_ball_center_px = selected_candidate["center"]

                            far_ball_bearing_deg = float(selected_candidate["bearing_deg"])
                            if far_ball_bearing_deg < -3.0:
                                far_ball_direction_text = "LEFT"
                            elif far_ball_bearing_deg > 3.0:
                                far_ball_direction_text = "RIGHT"
                            else:
                                far_ball_direction_text = "CENTER"
                            long_range_controller.far_target_last_bearing_deg = far_ball_bearing_deg
                            long_range_controller.far_target_last_direction_text = far_ball_direction_text

                            if long_range_controller.search_scan_phase == "far_approach":
                                if not long_range_controller.far_target_lock_active:
                                    print(
                                        f"[INFO] FAR target lock acquired at {far_ball_center_px}, conf={far_ball_best_conf:.2f}"
                                    )
                                long_range_controller.far_target_lock_active = True
                                long_range_controller.far_target_lock_px = far_ball_center_px
                                long_range_controller.far_target_lock_bearing_deg = far_ball_bearing_deg
                                long_range_controller.far_target_lock_conf = far_ball_best_conf

                        for cand in far_candidates:
                            is_selected = (
                                far_ball_detected
                                and cand["center"] == far_ball_center_px
                                and abs(cand["conf"] - far_ball_best_conf) < 1e-6
                            )
                            color = (0, 255, 255) if is_selected else (0, 255, 0)
                            thickness = 3 if is_selected else 2
                            label_text = f"far ball {cand['conf']:.2f}"
                            if is_selected:
                                label_text += " LOCK"
                            cv2.rectangle(
                                frame_csi0_bgr,
                                (cand["x0"], cand["y0"]),
                                (cand["x1"], cand["y1"]),
                                color,
                                thickness,
                            )
                            cv2.putText(
                                frame_csi0_bgr,
                                label_text,
                                (cand["x0"], max(18, cand["y0"] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                color,
                                2,
                            )

                        if far_ball_detected:
                            if far_ball_center_px is not None and far_ball_bearing_deg is not None:
                                cv2.circle(frame_csi0_bgr, far_ball_center_px, 6, (0, 255, 255), -1)
                                cv2.line(
                                    frame_csi0_bgr,
                                    (FRAME_W // 2, FRAME_H - 20),
                                    far_ball_center_px,
                                    (0, 255, 255),
                                    2,
                                )
                            print(
                                f"[INFO] Far ball detected on CSI0 "
                                f"(best conf={far_ball_best_conf:.2f}, bearing={far_ball_bearing_deg:+.1f}deg {far_ball_direction_text})"
                            )
                except Exception as exc:
                    if not csi0_preview_error_reported:
                        print(f"[WARNING] CSI0 preview capture failed, disabling preview: {exc}")
                        csi0_preview_error_reported = True
                    picam2_csi0 = None
                    far_detector_ready = False
                    long_range_controller.set_far_detector_ready(False)

            # Detect collection-like transition (ball was seen, then disappears)
            current_time = time.monotonic()
            
            # Track how long ball has been continuously detected
            if controller.ball_detected and not prev_ball_detected:
                # Ball just appeared
                ball_detected_since = current_time
                episode_min_ball_inlet_distance_m = float('inf')
            elif controller.ball_detected:
                # Track closest approach to inlet during this detection episode
                if controller.ball_pos_chassis is not None:
                    inlet_pos = controller.T_chassis_to_inlet[:3, 3]
                    distance_to_inlet = math.hypot(
                        controller.ball_pos_chassis[0] - inlet_pos[0],
                        controller.ball_pos_chassis[1] - inlet_pos[1],
                    )
                    episode_min_ball_inlet_distance_m = min(
                        episode_min_ball_inlet_distance_m,
                        distance_to_inlet,
                    )
            elif not controller.ball_detected and prev_ball_detected:
                # Ball just disappeared - check if it was detected long enough
                detection_duration = current_time - ball_detected_since
                in_cooldown = current_time < post_collect_cooldown_until
                near_inlet_capture = episode_min_ball_inlet_distance_m <= collect_near_inlet_distance_m
                ball_was_real = (detection_duration >= min_detection_duration_s) or near_inlet_capture
                
                if ball_was_real and not in_cooldown:
                    # Real ball was collected
                    balls_collected_count += 1
                    if abs(last_motion_linear_cmd) < 1e-3 and abs(last_motion_angular_cmd) < 1e-3:
                        last_motion_linear_cmd = min(0.03, chassis.max_linear_vel)
                        last_motion_angular_cmd = 0.0
                    post_collect_until = current_time + post_collect_duration_s
                    post_collect_cooldown_until = current_time + post_collect_cooldown_s
                    print(
                        f"[INFO] Ball collected #{balls_collected_count} "
                        f"(detected for {detection_duration:.2f}s, min inlet dist={episode_min_ball_inlet_distance_m*100:.1f}cm): "
                        f"keeping motors on for {post_collect_duration_s:.1f}s"
                    )
                elif not ball_was_real:
                    print(
                        f"[DEBUG] Ignoring brief/far detection "
                        f"({detection_duration:.2f}s < {min_detection_duration_s:.1f}s and "
                        f"min inlet dist {episode_min_ball_inlet_distance_m*100:.1f}cm > {collect_near_inlet_distance_m*100:.1f}cm)"
                    )
            
            prev_ball_detected = controller.ball_detected
            
            # Compute control commands
            planned_linear_vel = 0.0
            planned_angular_vel = 0.0
            
            if running and controller.ball_detected:
                planned_linear_vel, planned_angular_vel = controller.compute_control()
            
            # Determine actual commands to execute
            in_post_collect = time.monotonic() < post_collect_until
            if in_post_collect:
                linear_vel_cmd = last_motion_linear_cmd
                angular_vel_cmd = last_motion_angular_cmd
                inlet_active = True
                execute_motion = True
            elif manual_mode:
                # Manual mode: use keyboard commands
                linear_vel_cmd = manual_linear_vel
                angular_vel_cmd = math.radians(manual_angular_vel)
                inlet_active = user_inlet_enabled and controller.ball_detected
                execute_motion = True
            else:
                # Autonomous mode: planned inlet tracking + long-range search/approach fallback
                linear_vel_cmd, angular_vel_cmd, inlet_active, execute_motion = long_range_controller.compute_commands(
                    running=running,
                    ball_detected=controller.ball_detected,
                    user_inlet_enabled=user_inlet_enabled,
                    planned_linear_vel=planned_linear_vel,
                    planned_angular_vel=planned_angular_vel,
                    far_ball_detected=far_ball_detected,
                    far_ball_center_px=far_ball_center_px,
                    far_ball_bearing_deg=far_ball_bearing_deg,
                    far_ball_best_conf=far_ball_best_conf,
                    far_ball_direction_text=far_ball_direction_text,
                    chassis_max_linear_vel=chassis.max_linear_vel,
                )

            if execute_motion and not in_post_collect:
                last_motion_linear_cmd = linear_vel_cmd
                last_motion_angular_cmd = angular_vel_cmd
            
            # Apply commands to robot through motion worker thread
            with chassis.state_lock:
                chassis.linear_vel_cmd = linear_vel_cmd
                chassis.angular_vel_cmd = angular_vel_cmd
                chassis.inlet_enabled = inlet_active

            if execute_motion:
                chassis.run_event.set()
            else:
                chassis.run_event.clear()
            
            # Visualization
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            controller.draw_trajectory_on_frame(frame_bgr, ball_pixel_pos)
            
            # Status overlay
            # Control mode
            mode_text = "MANUAL" if manual_mode else ("AUTO:RUNNING" if running else "AUTO:STOPPED")
            mode_color = (0, 255, 255) if manual_mode else ((0, 255, 0) if running else (128, 128, 128))
            cv2.putText(frame_bgr, mode_text, (FRAME_W - 180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

            scan_text = f"Search: {'ON' if long_range_controller.search_scan_enabled else 'OFF'}"
            scan_color = (0, 200, 255) if long_range_controller.search_scan_enabled else (128, 128, 128)
            cv2.putText(frame_bgr, scan_text, (FRAME_W - 180, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, scan_color, 2)
            cv2.putText(frame_bgr, f"Phase: {long_range_controller.search_scan_phase}", (FRAME_W - 180, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            # Display current planner type
            planner_display = f"Planner: {controller.trajectory_planner.upper()}"
            cv2.putText(frame_bgr, planner_display, (FRAME_W - 180, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 100), 2)

            # Display collected ball count (session)
            count_display = f"Collected: {balls_collected_count}"
            cv2.putText(frame_bgr, count_display, (FRAME_W - 180, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Control signal display
            ctrl_y = FRAME_H - 130
            
            # Display planned trajectory commands (always calculated)
            cv2.putText(frame_bgr, "Planned (trajectory):", (10, ctrl_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            ctrl_y += 18
            cv2.putText(frame_bgr, f"  Lin: {planned_linear_vel:+.3f} m/s  Ang: {math.degrees(planned_angular_vel):+.1f} deg/s", (10, ctrl_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            ctrl_y += 22
            
            # Display actual commands being executed
            if manual_mode:
                cmd_label = "Executing (manual):"
                cmd_color = (0, 255, 255)
            elif in_post_collect:
                cmd_label = "Executing (collect+1s):"
                cmd_color = (0, 200, 255)
            elif execute_motion:
                cmd_label = "Executing (auto):"
                cmd_color = (0, 255, 0)
            else:
                cmd_label = "Executing (off):"
                cmd_color = (128, 128, 128)
            cv2.putText(frame_bgr, cmd_label, (10, ctrl_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cmd_color, 1)
            ctrl_y += 18
            cv2.putText(frame_bgr, f"  Lin: {linear_vel_cmd:+.3f} m/s  Ang: {math.degrees(angular_vel_cmd):+.1f} deg/s", (10, ctrl_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            ctrl_y += 20
            inlet_text = "ON" if inlet_active else "OFF"
            inlet_color = (0, 255, 0) if inlet_active else (128, 128, 128)
            cv2.putText(frame_bgr, f"Inlet: {inlet_text}", (10, ctrl_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, inlet_color, 1)
            
            # Compose single display window with side-by-side camera views
            display_frame = frame_bgr

            if picam2_csi0 is not None and frame_csi0_bgr is not None:
                try:
                    cv2.putText(
                        frame_csi0_bgr,
                        "Long Range (CSI0)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    if long_range_controller.search_scan_phase in ("far", "far_approach"):
                        far_status = "FAR DETECT: ON" if far_detector_ready else "FAR DETECT: OFF"
                        cv2.putText(
                            frame_csi0_bgr,
                            far_status,
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0) if far_detector_ready else (0, 0, 255),
                            2,
                        )
                        phase_label = "FAR SWEEP" if long_range_controller.search_scan_phase == "far" else "FAR APPROACH"
                        cv2.putText(
                            frame_csi0_bgr,
                            phase_label,
                            (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        if far_ball_detected:
                            cv2.putText(
                                frame_csi0_bgr,
                                f"BALL {far_ball_best_conf:.2f}",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
                            if far_ball_bearing_deg is not None:
                                cv2.putText(
                                    frame_csi0_bgr,
                                    f"DIR: {far_ball_direction_text} ({far_ball_bearing_deg:+.1f} deg)",
                                    (10, 125),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )
                        elif long_range_controller.search_scan_phase == "far_approach" and long_range_controller.far_target_last_bearing_deg is not None:
                            cv2.putText(
                                frame_csi0_bgr,
                                f"DIR: {long_range_controller.far_target_last_direction_text} ({long_range_controller.far_target_last_bearing_deg:+.1f} deg)",
                                (10, 125),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                    elif long_range_controller.search_scan_phase == "done" and long_range_controller.far_target_last_bearing_deg is not None:
                        cv2.putText(
                            frame_csi0_bgr,
                            f"LAST FAR TARGET: {long_range_controller.far_target_last_direction_text} ({long_range_controller.far_target_last_bearing_deg:+.1f} deg)",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )
                    cv2.putText(
                        frame_bgr,
                        f"Control View (CSI{args.camera})",
                        (10, FRAME_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    display_frame = np.hstack((frame_bgr, frame_csi0_bgr))
                    cv2.line(display_frame, (FRAME_W, 0), (FRAME_W, FRAME_H - 1), (255, 255, 255), 2)
                except Exception as exc:
                    if not csi0_preview_error_reported:
                        print(f"[WARNING] CSI0 preview capture failed, disabling preview: {exc}")
                        csi0_preview_error_reported = True
                    picam2_csi0 = None

            cv2.imshow("Ball Collection Control", display_frame)
            
            # Keyboard input
            key = cv2.waitKey(1)
            
            # Handle regular keys (ASCII)
            key_char = key & 0xFF
            if key_char == ord('q'):
                break
            elif key_char == ord(' '):
                running = not running
                long_range_controller.on_autonomy_toggled(running)
                print(f"Autonomous planning: {'RUNNING' if running else 'STOPPED'}")
            elif key_char == ord('m'):
                manual_mode = not manual_mode
                if manual_mode:
                    print("Switched to MANUAL control mode (arrow keys to drive)")
                    manual_linear_vel = 0.0
                    manual_angular_vel = 0.0
                    chassis.run_event.set()
                else:
                    print("Switched to AUTONOMOUS mode (robot stopped)")
                    chassis.run_event.clear()
                    # Stop robot when exiting manual mode
                    with chassis.state_lock:
                        chassis.linear_vel_cmd = 0.0
                        chassis.angular_vel_cmd = 0.0
                        chassis.inlet_enabled = False
            elif key_char == ord('i'):
                user_inlet_enabled = not user_inlet_enabled
                print(f"Inlet: {'ON' if user_inlet_enabled else 'OFF'}")
            elif key_char == ord('s'):
                search_scan_enabled = long_range_controller.toggle_search_scan()
                print(
                    f"Search scan mode: {'ON' if search_scan_enabled else 'OFF'} "
                    f"(sweep angle={long_range_controller.search_scan_angle_deg:.1f}deg)"
                )
            elif key_char == ord('1'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_BEZIER)
            elif key_char == ord('2'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_SHAPED)
            elif key_char == ord('3'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_DIRECT)
            elif key_char == ord('4'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_ARC)
            elif key_char == ord('r'):
                balls_collected_count = 0
                print("Ball counter reset to 0")
            
            # Manual control with arrow keys (only active in manual mode)
            # Arrow key codes on Linux: UP=82, DOWN=84, LEFT=81, RIGHT=83 (when used with 0xFF00 mask)
            if manual_mode and key != -1:
                # Check for arrow keys (extended codes)
                if key == 82 or key == 2490368:  # UP arrow
                    manual_linear_vel = min(chassis.max_linear_vel, manual_linear_vel + key_linear_step)
                    print(f"Manual: linear velocity = {manual_linear_vel:.3f} m/s")
                elif key == 84 or key == 2621440:  # DOWN arrow
                    manual_linear_vel = max(-chassis.max_linear_vel, manual_linear_vel - key_linear_step)
                    print(f"Manual: linear velocity = {manual_linear_vel:.3f} m/s")
                elif key == 81 or key == 2424832:  # LEFT arrow
                    manual_angular_vel = min(chassis.max_angular_vel_deg, manual_angular_vel + key_angular_step)
                    print(f"Manual: angular velocity = {manual_angular_vel:.1f} deg/s")
                elif key == 83 or key == 2555904:  # RIGHT arrow
                    manual_angular_vel = max(-chassis.max_angular_vel_deg, manual_angular_vel - key_angular_step)
                    print(f"Manual: angular velocity = {manual_angular_vel:.1f} deg/s")
            
            # Maintain control rate
            elapsed = time.monotonic() - loop_start
            sleep_time = controller.control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        print("Shutting down...")
        chassis.run_event.clear()
        chassis.stop_event.set()
        # Stop robot motion
        with chassis.state_lock:
            chassis.linear_vel_cmd = 0.0
            chassis.angular_vel_cmd = 0.0
            chassis.inlet_enabled = False
        time.sleep(0.1)
        if 'motion_thread' in locals() and motion_thread.is_alive():
            motion_thread.join(timeout=1.0)
        picam2.stop()
        if 'picam2_csi0' in locals() and picam2_csi0 is not None:
            picam2_csi0.stop()
        cv2.destroyAllWindows()
        portHandler.closePort()
        print("Done")


if __name__ == "__main__":
    main()
