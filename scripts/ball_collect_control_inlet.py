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

# Constants from ball_finder
CALIB_FILE = "../data/camera_intrinsics_640x480_20cmfocus.npz"
FRAME_W, FRAME_H = 640, 480
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
TARGET_LABELS = {"sports ball"}
CONF_MIN = 0.30

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
        max_linear_vel=0.15,  # Slower for visual servoing
        max_angular_vel_deg=30,
        max_inlet_vel=3.0
    )
    
    # Transformation matrices (ADJUST THESE FOR YOUR ROBOT)
    # Example: Camera mounted 10cm forward, 0cm lateral, 26cm up from chassis center
    T_chassis_to_camera = np.array([
        [1, 0, 0, 0.10],  # Camera 10cm forward
        [0, 1, 0, 0.00],  # Camera at center laterally
        [0, 0, 1, 0.26],  # Camera 26cm above ground
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    # Inlet in front of camera (so it's visible in camera view)
    T_chassis_to_inlet = np.array([
        [1, 0, 0, 0.05],  # Inlet 5cm forward (5cm behind of camera)
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
    print("  'q' = quit")
    print("  1-4 = select trajectory planner")
    print("    1: Bézier curve (default)")
    print("    2: Shaped approach")
    print("    3: Direct line")
    print("    4: Arc")
    print("Manual keyboard control (when in manual mode):")
    print("  UP/DOWN arrows = linear velocity +/- 0.01 m/s")
    print("  LEFT/RIGHT arrows = angular velocity +/- 1 deg/s")
    print("Starting control loop...")

    chassis.stop_event.clear()
    chassis.run_event.clear()
    motion_thread = threading.Thread(target=chassis.motionControlWorker, daemon=True)
    motion_thread.start()
    
    # Control loop state
    running = False
    user_inlet_enabled = True  # User's inlet preference
    manual_mode = False  # Manual keyboard control vs autonomous
    
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
    last_motion_linear_cmd = 0.0
    last_motion_angular_cmd = 0.0
    
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

            # Detect collection-like transition (ball was seen, then disappears)
            current_time = time.monotonic()
            
            # Track how long ball has been continuously detected
            if controller.ball_detected and not prev_ball_detected:
                # Ball just appeared
                ball_detected_since = current_time
            elif not controller.ball_detected and prev_ball_detected:
                # Ball just disappeared - check if it was detected long enough
                detection_duration = current_time - ball_detected_since
                in_cooldown = current_time < post_collect_cooldown_until
                ball_was_real = detection_duration >= min_detection_duration_s
                
                if ball_was_real and not in_cooldown:
                    # Real ball was collected
                    if abs(last_motion_linear_cmd) < 1e-3 and abs(last_motion_angular_cmd) < 1e-3:
                        last_motion_linear_cmd = min(0.03, chassis.max_linear_vel)
                        last_motion_angular_cmd = 0.0
                    post_collect_until = current_time + post_collect_duration_s
                    post_collect_cooldown_until = current_time + post_collect_cooldown_s
                    print(f"[INFO] Ball collected (detected for {detection_duration:.2f}s): keeping motors on for {post_collect_duration_s:.1f}s")
                elif not ball_was_real:
                    print(f"[DEBUG] Ignoring brief detection ({detection_duration:.2f}s < {min_detection_duration_s:.1f}s threshold)")
            
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
                # Autonomous mode: execute planned trajectory when running
                if running and controller.ball_detected:
                    linear_vel_cmd = planned_linear_vel
                    angular_vel_cmd = planned_angular_vel
                    inlet_active = user_inlet_enabled and controller.ball_detected
                    execute_motion = True
                else:
                    linear_vel_cmd = 0.0
                    angular_vel_cmd = 0.0
                    inlet_active = False
                    execute_motion = False

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
            
            # Display current planner type
            planner_display = f"Planner: {controller.trajectory_planner.upper()}"
            cv2.putText(frame_bgr, planner_display, (FRAME_W - 180, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 100), 2)
            
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
            
            cv2.imshow("Ball Collection Control", frame_bgr)
            
            # Keyboard input
            key = cv2.waitKey(1)
            
            # Handle regular keys (ASCII)
            key_char = key & 0xFF
            if key_char == ord('q'):
                break
            elif key_char == ord(' '):
                running = not running
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
            elif key_char == ord('1'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_BEZIER)
            elif key_char == ord('2'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_SHAPED)
            elif key_char == ord('3'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_DIRECT)
            elif key_char == ord('4'):
                controller.set_trajectory_planner(TRAJECTORY_PLANNER_ARC)
            
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
        cv2.destroyAllWindows()
        portHandler.closePort()
        print("Done")


if __name__ == "__main__":
    main()
