#!/usr/bin/env python3
import time
import sys
import math
import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# User-configurable defaults
# =========================

# --- Calibration file (from your previous calibration script) ---
CALIB_FILE = "../data/camera_intrinsics_640x480_20cmfocus.npz"

# --- Camera capture ---
FRAME_W = 640
FRAME_H = 480

# --- Ball parameters ---
BALL_DIAMETER_M = 0.067   # tennis ball ~ 6.7 cm
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0

# --- Camera mounting geometry ---
CAMERA_HEIGHT_M = 0.236    # camera optical center height above floor [m]
CAMERA_TILT_DEG = 9.0    # positive = camera pitched downward from horizontal [deg]

# --- Detection thresholds (tennis-ball-ish yellow/green) ---
# You may need to tune these for your lighting / white balance.
HSV_LOWER = np.array([78, 58, 40], dtype=np.uint8)
HSV_UPPER = np.array([89, 255, 255], dtype=np.uint8)

# --- Detection filtering ---
MIN_RADIUS_PX = 6
MIN_CONTOUR_AREA = 80
MIN_CIRCULARITY = 0.45

# --- Display / behavior ---
SHOW_MASK = True
USE_PLANE_METHOD = True          # uses camera height + tilt + ball radius
USE_SIZE_METHOD = True           # uses apparent ball size + known diameter
SMOOTHING_ALPHA = 0.35           # 0~1, larger = faster / noisier (EMA on chosen output)
PREFER_METHOD = "plane"          # "plane" or "size"

# =========================
# Utility functions
# =========================

def load_calibration(npz_path: str):
    data = np.load(npz_path)
    K = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)
    return K, dist

def undistort_pixel_to_ray(u, v, K, dist):
    """
    Convert pixel coordinate (u,v) to a 3D ray direction in the camera frame.
    Returns:
      ray (unnormalized): [x, y, 1]
      ray_unit: normalized direction
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)  # shape (1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=None)             # normalized coordinates
    x = und[0, 0, 0]
    y = und[0, 0, 1]
    ray = np.array([x, y, 1.0], dtype=np.float64)
    ray_unit = ray / np.linalg.norm(ray)
    return ray, ray_unit

def intersect_ray_with_ball_center_plane(ray, camera_height_m, tilt_deg, ball_radius_m):
    """
    Intersect camera ray with the plane containing centers of balls resting on the ground.

    Camera frame convention (OpenCV):
      x right, y down, z forward

    If camera tilt = 0 (horizontal), ground plane is y = H.
    For downward tilt theta, the "ball-center plane" becomes:
        y*cos(theta) + z*sin(theta) = H - R
    where H is camera height above ground, R is ball radius.

    Returns:
      p_cam (3,) center position in camera frame [m], or None if invalid.
    """
    H = float(camera_height_m)
    R = float(ball_radius_m)
    theta = math.radians(float(tilt_deg))

    if H <= R:
        return None  # impossible geometry: camera below ball center height plane

    # Plane normal (in camera frame) and RHS
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    rhs = H - R  # plane offset for BALL CENTER plane (not the floor plane)

    denom = float(np.dot(n, ray))
    if denom <= 1e-8:
        return None  # ray does not intersect in front / near horizon

    t = rhs / denom
    if t <= 0:
        return None

    p_cam = t * ray  # ray = [x,y,1] normalized-coordinate ray (not unit), but valid for scaling intersection
    return p_cam

def estimate_center_from_apparent_ball_size(u, v, radius_px, K, dist, ball_radius_m):
    """
    Estimate sphere center distance from apparent angular radius.
    Uses center pixel and a boundary pixel (to the right by radius_px).

    Geometry:
      angular radius alpha satisfies sin(alpha) = R / d_center
      => d_center = R / sin(alpha)

    Returns:
      p_cam (3,) estimated center position in camera frame [m], or None.
    """
    if radius_px <= 1.0:
        return None

    # Center ray
    _, rc = undistort_pixel_to_ray(u, v, K, dist)

    # Boundary point (right side of detected image circle)
    ue = u + radius_px
    ve = v
    if ue < 0 or ue >= FRAME_W:
        ue = max(0, min(FRAME_W - 1, ue))
    _, re = undistort_pixel_to_ray(ue, ve, K, dist)

    dot = float(np.clip(np.dot(rc, re), -1.0, 1.0))
    alpha = math.acos(dot)  # angular radius

    if alpha < 1e-6:
        return None

    s = math.sin(alpha)
    if s <= 1e-8:
        return None

    d_center = ball_radius_m / s  # distance from camera center to ball center
    p_cam = d_center * rc
    return p_cam

def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))

def detect_tennis_ball(frame_bgr):
    """
    Returns best detected ball candidate:
      dict with keys {center_u, center_v, radius_px, contour, mask}, or None
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    # Clean mask
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        circ = contour_circularity(cnt)
        if circ < MIN_CIRCULARITY:
            continue

        (u, v), radius = cv2.minEnclosingCircle(cnt)
        if radius < MIN_RADIUS_PX:
            continue

        # Score: prefer larger and more circular blobs
        score = area * (0.5 + 0.5 * circ)

        if score > best_score:
            best_score = score
            best = {
                "center_u": float(u),
                "center_v": float(v),
                "radius_px": float(radius),
                "contour": cnt,
                "mask": mask,
                "area": float(area),
                "circularity": float(circ),
            }

    if best is None:
        return None, mask
    return best, mask

def fmt_vec_cm(p):
    if p is None:
        return "None"
    return f"[{p[0]*100: .1f}, {p[1]*100: .1f}, {p[2]*100: .1f}] cm"

# =========================
# Main
# =========================

def main(camera_num: int = 0):
    # Load calibration
    K, dist = load_calibration(CALIB_FILE)
    print("Loaded calibration:")
    print("K=\n", K)
    print("dist=", dist.ravel())

    print("\nCamera frame convention (OpenCV): x-right, y-down, z-forward")
    print(f"Using camera: CSI {camera_num}")
    print(f"Camera height = {CAMERA_HEIGHT_M:.3f} m")
    print(f"Camera tilt   = {CAMERA_TILT_DEG:.2f} deg (downward positive)")
    print(f"Ball diameter = {BALL_DIAMETER_M:.3f} m")
    print("\nKeys: q=quit")

    # Start camera
    picam2 = Picamera2(camera_num=camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    ema_p = None  # smoothed chosen estimate

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        vis = frame_bgr.copy()

        det, mask = detect_tennis_ball(frame_bgr)

        p_plane = None
        p_size = None
        p_chosen = None

        if det is not None:
            u = det["center_u"]
            v = det["center_v"]
            r_px = det["radius_px"]

            # Draw detection
            cv2.circle(vis, (int(round(u)), int(round(v))), int(round(r_px)), (0, 255, 0), 2)
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)

            # Ray through center pixel
            ray, ray_unit = undistort_pixel_to_ray(u, v, K, dist)

            if USE_PLANE_METHOD:
                p_plane = intersect_ray_with_ball_center_plane(
                    ray=ray,
                    camera_height_m=CAMERA_HEIGHT_M,
                    tilt_deg=CAMERA_TILT_DEG,
                    ball_radius_m=BALL_RADIUS_M
                )

            if USE_SIZE_METHOD:
                p_size = estimate_center_from_apparent_ball_size(
                    u=u, v=v, radius_px=r_px,
                    K=K, dist=dist,
                    ball_radius_m=BALL_RADIUS_M
                )

            # Choose output
            if PREFER_METHOD.lower() == "plane":
                p_chosen = p_plane if p_plane is not None else p_size
            else:
                p_chosen = p_size if p_size is not None else p_plane

            # EMA smoothing
            if p_chosen is not None:
                if ema_p is None:
                    ema_p = p_chosen.copy()
                else:
                    ema_p = (1.0 - SMOOTHING_ALPHA) * ema_p + SMOOTHING_ALPHA * p_chosen
            else:
                # Keep last smoothed value, or you can set ema_p=None if you prefer
                pass

            # Overlay text
            cv2.putText(vis, f"u,v=({u:.1f},{v:.1f})  r={r_px:.1f}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(vis, f"area={det['area']:.0f} circ={det['circularity']:.2f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.putText(vis, f"Plane est: {fmt_vec_cm(p_plane)}",
                        (10, FRAME_H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.putText(vis, f"Size  est: {fmt_vec_cm(p_size)}",
                        (10, FRAME_H - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2)
            cv2.putText(vis, f"Chosen(smooth): {fmt_vec_cm(ema_p)}",
                        (10, FRAME_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

            # Optional: print to terminal at low rate (uncomment if desired)
            # print("Plane:", p_plane, "Size:", p_size, "Chosen:", ema_p)

        else:
            cv2.putText(vis, "Ball not detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis, "Tune HSV_LOWER / HSV_UPPER for your lighting",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Display geometry info
        cv2.putText(vis, f"H={CAMERA_HEIGHT_M:.3f}m tilt={CAMERA_TILT_DEG:.1f}deg D={BALL_DIAMETER_M:.3f}m",
                    (10, FRAME_H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)

        # Show image
        cv2.imshow("Ball Position Estimation (camera frame)", vis)
        if SHOW_MASK:
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_num = 0
    if len(sys.argv) > 1:
        try:
            camera_num = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [camera_num]")
            print("  camera_num: 0 or 1 (default: 0)")
            sys.exit(1)
    main(camera_num=camera_num)