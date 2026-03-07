#!/usr/bin/env python3
import time
import math
import argparse
import cv2
import numpy as np
from picamera2 import Picamera2

# NOT WORKING AT THE MOMENT

# =========================
# User defaults (edit these)
# =========================
FRAME_W, FRAME_H = 640, 480

# Expected ball radius in pixels (wide range if distance varies)
R_MIN_PX = 6
R_MAX_PX = 120

# Learning settings
H_COVERAGE = 0.90         # smallest hue interval covering this fraction of inside pixels
S_LO_P = 10               # percentile for S low bound (inside)
V_LO_P = 10               # percentile for V low bound (inside)
S_HI_P = 99               # percentile for S high bound (inside)
V_HI_P = 99               # percentile for V high bound (inside)

H_MARGIN = 4              # expand hue interval by +/- this many hue units
S_MARGIN = 10             # expand S bounds
V_MARGIN = 10             # expand V bounds

# Validation thresholds (strict to avoid learning from non-ball regions)
MIN_INSIDE_COVERAGE = 0.80  # how much of the circle must be explained by HSV
MAX_RING_LEAKAGE = 0.08     # how much background ring can be falsely included
MIN_COLOR_UNIFORMITY = 0.65 # inside region should be relatively uniform (not noisy)

# EMA update rate
EMA_ALPHA = 0.12            # 0.05 (slow) .. 0.2 (faster)

# Ring sampling relative to circle radius
RING_IN_SCALE = 1.25
RING_OUT_SCALE = 1.80

# HoughCircles params (tune if needed)
HOUGH_DP = 1.0
HOUGH_MIN_DIST = 40
HOUGH_PARAM1 = 50    # Canny high threshold (lower = more sensitive)
HOUGH_PARAM2 = 15    # accumulator threshold (lower -> more detections)
# =========================

# Debug mode
DEBUG_MODE = True    # Show additional debug info

# Center ROI constraint (fraction of frame, 0.5 = center 50%)
ROI_CENTER_FRACTION = 0.35  # Only consider circles in center 65% of frame
SHOW_ROI_BOX = True          # Show ROI boundary on screen


def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(round(x)))))


def circular_shortest_interval(h_vals, coverage=0.90):
    """
    Find the shortest hue interval [h_lo, h_hi] (mod 180) containing `coverage` fraction of samples.
    Returns (h_lo, h_hi, wraps) where wraps=True means interval crosses 0.
    """
    h = np.sort(h_vals.astype(np.int32))
    n = len(h)
    if n < 10:
        return None

    k = max(1, int(math.ceil(coverage * n)))
    # Duplicate with +180 to handle wrap-around windows
    h2 = np.concatenate([h, h + 180])

    best_len = 1e9
    best_lo = None
    best_hi = None
    for i in range(0, n):
        j = i + k - 1
        if j >= i + n:
            break
        length = h2[j] - h2[i]
        if length < best_len:
            best_len = length
            best_lo = h2[i]
            best_hi = h2[j]

    if best_lo is None:
        return None

    lo = best_lo % 180
    hi = best_hi % 180
    wraps = (best_hi - best_lo) > ((hi - lo) % 180)  # interval likely wraps around 0
    # A more direct wrap check:
    wraps = (best_lo < 180) and (best_hi >= 180) and (hi < lo)

    return int(lo), int(hi), wraps


def build_circle_mask(shape_hw, cx, cy, r):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
    return mask


def build_ring_mask(shape_hw, cx, cy, r_in, r_out):
    h, w = shape_hw
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring, (cx, cy), r_out, 255, thickness=-1)
    cv2.circle(ring, (cx, cy), r_in, 0, thickness=-1)
    return ring


def hsv_inrange_wrap(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    """
    Apply HSV threshold with hue wrap-around support.
    h in [0..179], s/v in [0..255]
    """
    s_lo = clamp_int(s_lo, 0, 255)
    s_hi = clamp_int(s_hi, 0, 255)
    v_lo = clamp_int(v_lo, 0, 255)
    v_hi = clamp_int(v_hi, 0, 255)
    h_lo = clamp_int(h_lo, 0, 179)
    h_hi = clamp_int(h_hi, 0, 179)

    if h_lo <= h_hi:
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)
    else:
        # wrap: [0..h_hi] U [h_lo..179]
        lower1 = np.array([0, s_lo, v_lo], dtype=np.uint8)
        upper1 = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        lower2 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper2 = np.array([179, s_hi, v_hi], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))


def compute_gradient_score(gray, cx, cy, r):
    """
    Score edge support along the circle boundary using gradient magnitude.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Sample points around the circle
    angles = np.linspace(0, 2*np.pi, 60, endpoint=False)
    xs = (cx + r * np.cos(angles)).astype(np.int32)
    ys = (cy + r * np.sin(angles)).astype(np.int32)

    h, w = gray.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 10:
        return 0.0

    return float(np.mean(mag[ys, xs]))


def check_color_uniformity(hsv, mask):
    """
    Check if the masked region has relatively uniform color (not noisy/mixed).
    Returns a score [0..1] where higher is more uniform.
    """
    h_vals = hsv[:, :, 0][mask > 0]
    s_vals = hsv[:, :, 1][mask > 0]
    v_vals = hsv[:, :, 2][mask > 0]
    
    if len(h_vals) < 50:
        return 0.0
    
    # Compute coefficient of variation (std/mean) for each channel
    h_cv = np.std(h_vals) / (np.mean(h_vals) + 1.0)
    s_cv = np.std(s_vals) / (np.mean(s_vals) + 1.0)
    v_cv = np.std(v_vals) / (np.mean(v_vals) + 1.0)
    
    # Convert to uniformity score: lower CV = higher uniformity
    avg_cv = (h_cv + s_cv + v_cv) / 3.0
    uniformity = max(0.0, 1.0 - avg_cv)
    
    return float(uniformity)


def check_ball_color(hsv, mask):
    """
    Check if the masked region has tennis ball color characteristics.
    Returns True if it looks like a tennis ball (yellow-green, high saturation).
    """
    h_vals = hsv[:, :, 0][mask > 0]
    s_vals = hsv[:, :, 1][mask > 0]
    v_vals = hsv[:, :, 2][mask > 0]
    
    if len(h_vals) < 50:
        return False, "Not enough pixels"
    
    # Tennis ball: yellow-green hue (15-50), high saturation (>60), decent brightness (>50)
    h_med = float(np.median(h_vals))
    s_med = float(np.median(s_vals))
    v_med = float(np.median(v_vals))
    
    # Reject gray/black objects (wheels, screws)
    if s_med < 50:
        return False, f"Low saturation S={s_med:.0f}<50 (gray/black object)"
    
    # Reject very dark objects
    if v_med < 30:
        return False, f"Too dark V={v_med:.0f}<30"
    
    # Tennis ball hue range (yellow-green): roughly 15-50 in OpenCV
    # Allow some tolerance for lighting conditions
    if not (10 <= h_med <= 65):
        return False, f"Wrong hue H={h_med:.0f} (not yellow-green)"
    
    # Check saturation distribution - should be consistently high
    s_25 = float(np.percentile(s_vals, 25))
    if s_25 < 40:
        return False, f"Low sat variation S_25={s_25:.0f}<40"
    
    return True, f"H={h_med:.0f} S={s_med:.0f} V={v_med:.0f}"


def is_in_center_roi(cx, cy, frame_w, frame_h, roi_fraction):
    """
    Check if a point (cx, cy) is within the center ROI.
    roi_fraction: 0.5 means center 50% of frame, 1.0 means entire frame
    """
    margin_x = int(frame_w * (1.0 - roi_fraction) / 2)
    margin_y = int(frame_h * (1.0 - roi_fraction) / 2)
    
    x_min = margin_x
    x_max = frame_w - margin_x
    y_min = margin_y
    y_max = frame_h - margin_y
    
    return x_min <= cx <= x_max and y_min <= cy <= y_max


def get_roi_bounds(frame_w, frame_h, roi_fraction):
    """
    Get the ROI bounding box coordinates.
    Returns (x_min, y_min, x_max, y_max)
    """
    margin_x = int(frame_w * (1.0 - roi_fraction) / 2)
    margin_y = int(frame_h * (1.0 - roi_fraction) / 2)
    
    x_min = margin_x
    x_max = frame_w - margin_x
    y_min = margin_y
    y_max = frame_h - margin_y
    
    return x_min, y_min, x_max, y_max


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Adaptive HSV tuner for tennis ball detection")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        choices=[0, 1],
        help="Camera source: 0 (CSI 0) or 1 (CSI 1) (default: 0)"
    )
    args = parser.parse_args()

    # Initial (very broad) HSV bounds; will adapt
    hsv_bounds = {
        "h_lo": 25, "h_hi": 140,
        "s_lo": 30, "s_hi": 255,
        "v_lo": 30, "v_hi": 255
    }

    picam2 = Picamera2(camera_num=args.camera)
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    last_circle = None  # (cx,cy,r)
    stable_count = 0
    locked = False      # Ball detection locked state
    ball_confirmed = False  # We've confirmed a valid ball is detected
    
    # Dynamic parameters (can be adjusted with keyboard)
    hough_param1 = HOUGH_PARAM1
    hough_param2 = HOUGH_PARAM2
    debug_mode = DEBUG_MODE
    show_rejections = False
    roi_fraction = ROI_CENTER_FRACTION
    show_roi = SHOW_ROI_BOX
    
    last_rejection_reasons = []  # Store rejection reasons from last frame

    print("Adaptive HSV tuner running.")
    print("  1. Point camera at a tennis ball")
    print("  2. Wait for detection (green circle)")
    print("  3. Press SPACE to lock/unlock detection")
    print("  4. Press 'q' to quit and save bounds")
    print("\nControls:")
    print("  W/S: Adjust HOUGH_PARAM2 (accumulator threshold)")
    print("  +/-: Adjust HOUGH_PARAM1 (Canny threshold)")
    print("  A/E (or Left/Right): Decrease/Increase ROI size (center region)")
    print("  t: Toggle debug mode")
    print("  b: Toggle ROI box display")
    print("  r: Show rejection reasons\n")

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 2.0)

        # --- 1) Detect circle candidates (non-color) ---
        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=hough_param1,
            param2=hough_param2,
            minRadius=R_MIN_PX,
            maxRadius=R_MAX_PX
        )
        
        # Debug: Show edge detection used by HoughCircles
        if debug_mode:
            edges = cv2.Canny(gray_blur, int(hough_param1 / 2), hough_param1)
            cv2.imshow("Edges (Canny)", edges)
        elif cv2.getWindowProperty("Edges (Canny)", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("Edges (Canny)")

        best = None
        best_score = -1e9
        num_candidates = 0
        num_ball_colored = 0
        num_in_roi = 0
        rejection_reasons = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype(np.int32)
            num_candidates = len(circles)

            for (cx, cy, r) in circles:
                if r < R_MIN_PX or r > R_MAX_PX:
                    rejection_reasons.append(f"Circle at ({cx},{cy}) r={r}: Size out of range")
                    continue

                # Check if circle is in center ROI
                if not is_in_center_roi(cx, cy, FRAME_W, FRAME_H, roi_fraction):
                    rejection_reasons.append(f"Circle at ({cx},{cy}) r={r}: Outside center ROI")
                    continue
                
                num_in_roi += 1

                # Pre-filter: Check if this circle has ball-like color
                circle_mask_check = build_circle_mask((FRAME_H, FRAME_W), cx, cy, r)
                is_ball_color, color_reason = check_ball_color(hsv, circle_mask_check)
                
                if not is_ball_color:
                    # Skip this circle - doesn't have ball color
                    rejection_reasons.append(f"Circle at ({cx},{cy}) r={r}: {color_reason}")
                    continue
                
                num_ball_colored += 1

                # Edge score
                edge_score = compute_gradient_score(gray_blur, cx, cy, r)

                # Temporal prior: prefer near last circle
                prior = 0.0
                if last_circle is not None:
                    lx, ly, lr = last_circle
                    d = math.hypot(cx - lx, cy - ly)
                    dr = abs(r - lr)
                    prior = -0.02 * d - 0.05 * dr  # small penalty for jumping

                score = edge_score + prior
                if score > best_score:
                    best_score = score
                    best = (cx, cy, r, edge_score, color_reason)
        
        last_rejection_reasons = rejection_reasons

        vis = frame_bgr.copy()
        
        # Draw ROI box if enabled
        if show_roi:
            x_min, y_min, x_max, y_max = get_roi_bounds(FRAME_W, FRAME_H, roi_fraction)
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (255, 165, 0), 2)  # Orange box
            roi_text = f"ROI: {int(roi_fraction*100)}%"
            cv2.putText(vis, roi_text, (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        # Debug: Show detection status and draw circles
        if debug_mode:
            num_rejected = num_candidates - num_ball_colored
            debug_txt = f"Circles: {num_candidates} total | {num_in_roi} in ROI | {num_ball_colored} ball-like | {num_rejected} rejected | P1:{hough_param1} P2:{hough_param2}"
            cv2.putText(vis, debug_txt, (10, FRAME_H - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw all circles with color coding
            if circles is not None:
                for (cx, cy, r) in circles:
                    # Check ROI first
                    in_roi = is_in_center_roi(cx, cy, FRAME_W, FRAME_H, roi_fraction)
                    
                    if not in_roi:
                        # Outside ROI - dark red
                        cv2.circle(vis, (cx, cy), r, (0, 0, 100), 1)
                        cv2.circle(vis, (cx, cy), 2, (0, 0, 150), -1)
                        continue
                    
                    circle_mask_check = build_circle_mask((FRAME_H, FRAME_W), cx, cy, r)
                    is_ball_color, color_reason = check_ball_color(hsv, circle_mask_check)
                    if is_ball_color:
                        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 1)  # Yellow for ball-like
                    else:
                        cv2.circle(vis, (cx, cy), r, (80, 80, 80), 1)  # Gray for rejected
                        cv2.line(vis, (cx-8, cy-8), (cx+8, cy+8), (0, 0, 200), 2)  # Red X
                        cv2.line(vis, (cx-8, cy+8), (cx+8, cy-8), (0, 0, 200), 2)

        # --- 2) If we have a candidate, learn HSV bounds from it ---
        updated_this_frame = False
        if best is not None:
            cx, cy, r, edge_score, color_reason = best

            # Stability heuristic (optional)
            if last_circle is not None:
                lx, ly, lr = last_circle
                if math.hypot(cx - lx, cy - ly) < 25 and abs(r - lr) < 15:
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0
            last_circle = (cx, cy, r)

            circle_mask = build_circle_mask((FRAME_H, FRAME_W), cx, cy, r)
            ring_mask = build_ring_mask(
                (FRAME_H, FRAME_W),
                cx, cy,
                int(r * RING_IN_SCALE),
                int(r * RING_OUT_SCALE)
            )

            # Extract HSV samples
            h_in = hsv[:, :, 0][circle_mask > 0]
            s_in = hsv[:, :, 1][circle_mask > 0]
            v_in = hsv[:, :, 2][circle_mask > 0]

            h_out = hsv[:, :, 0][ring_mask > 0]
            s_out = hsv[:, :, 1][ring_mask > 0]
            v_out = hsv[:, :, 2][ring_mask > 0]

            if len(h_in) > 200 and len(h_out) > 200:
                # Hue interval (robust + wrap)
                hue_int = circular_shortest_interval(h_in, coverage=H_COVERAGE)

                if hue_int is not None:
                    h_lo, h_hi, wraps = hue_int
                    # Expand hue interval a bit
                    h_lo = (h_lo - H_MARGIN) % 180
                    h_hi = (h_hi + H_MARGIN) % 180

                    # S/V bounds via percentiles inside circle
                    s_lo = int(max(0, np.percentile(s_in, S_LO_P) - S_MARGIN))
                    v_lo = int(max(0, np.percentile(v_in, V_LO_P) - V_MARGIN))
                    s_hi = int(min(255, np.percentile(s_in, S_HI_P) + S_MARGIN))
                    v_hi = int(min(255, np.percentile(v_in, V_HI_P) + V_MARGIN))

                    # --- 3) Validate separation (inside coverage vs ring leakage) ---
                    mask_candidate = hsv_inrange_wrap(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)

                    inside_pass = np.count_nonzero(mask_candidate[circle_mask > 0]) / max(1, np.count_nonzero(circle_mask))
                    ring_pass = np.count_nonzero(mask_candidate[ring_mask > 0]) / max(1, np.count_nonzero(ring_mask))
                    
                    # Check color uniformity inside the circle
                    uniformity = check_color_uniformity(hsv, circle_mask)

                    # Extra sanity: avoid learning from low-saturation or too-dark region
                    s_med_in = float(np.median(s_in))
                    v_med_in = float(np.median(v_in))

                    # Validation: strict requirements to ensure ball-only detection
                    passes_validation = (
                        inside_pass >= MIN_INSIDE_COVERAGE and 
                        ring_pass <= MAX_RING_LEAKAGE and 
                        s_med_in >= 30 and 
                        v_med_in >= 40 and
                        uniformity >= MIN_COLOR_UNIFORMITY and
                        stable_count >= 2  # Must be stable for at least 2 frames (relaxed)
                    )

                    if locked and passes_validation:
                        ball_confirmed = True
                        # --- 4) EMA update of bounds (only when locked and valid) ---
                        def ema(old, new):
                            return (1.0 - EMA_ALPHA) * old + EMA_ALPHA * new

                        hsv_bounds["h_lo"] = int(round(ema(hsv_bounds["h_lo"], h_lo)))
                        hsv_bounds["h_hi"] = int(round(ema(hsv_bounds["h_hi"], h_hi)))
                        hsv_bounds["s_lo"] = int(round(ema(hsv_bounds["s_lo"], s_lo)))
                        hsv_bounds["s_hi"] = int(round(ema(hsv_bounds["s_hi"], s_hi)))
                        hsv_bounds["v_lo"] = int(round(ema(hsv_bounds["v_lo"], v_lo)))
                        hsv_bounds["v_hi"] = int(round(ema(hsv_bounds["v_hi"], v_hi)))

                        updated_this_frame = True

                    # Draw diagnostics
                    circle_color = (0, 255, 0) if passes_validation else (0, 0, 255)
                    circle_thickness = 3 if ball_confirmed else 2
                    cv2.circle(vis, (cx, cy), r, circle_color, circle_thickness)
                    cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
                    
                    # Status text
                    status = "LOCKED" if locked else "UNLOCKED"
                    status_color = (0, 255, 0) if locked else (0, 165, 255)
                    cv2.putText(vis, f"Status: {status} | Ball: {'YES' if ball_confirmed else 'NO'} | {color_reason}",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
                    
                    cv2.putText(vis, f"Edge={edge_score:.1f} Stable={stable_count} Uniform={uniformity:.2f}",
                                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    
                    cv2.putText(vis, f"In={inside_pass:.2f} Ring={ring_pass:.2f} Valid={passes_validation}",
                                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 
                                (0, 255, 0) if passes_validation else (0, 0, 255), 1)
                    
                    # Debug: show why validation might be failing
                    if debug_mode and not passes_validation:
                        fail_reasons = []
                        if inside_pass < MIN_INSIDE_COVERAGE: fail_reasons.append(f"Inside:{inside_pass:.2f}<{MIN_INSIDE_COVERAGE}")
                        if ring_pass > MAX_RING_LEAKAGE: fail_reasons.append(f"Ring:{ring_pass:.2f}>{MAX_RING_LEAKAGE}")
                        if s_med_in < 30: fail_reasons.append(f"S:{s_med_in:.0f}<30")
                        if v_med_in < 40: fail_reasons.append(f"V:{v_med_in:.0f}<40")
                        if uniformity < MIN_COLOR_UNIFORMITY: fail_reasons.append(f"Unif:{uniformity:.2f}<{MIN_COLOR_UNIFORMITY}")
                        if stable_count < 2: fail_reasons.append(f"Stab:{stable_count}<2")
                        
                        fail_txt = " | ".join(fail_reasons)
                        cv2.putText(vis, f"Fail: {fail_txt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        else:
            # No circle detected or all circles rejected
            if debug_mode:
                if num_candidates > 0:
                    cv2.putText(vis, f"Found {num_candidates} circles but all rejected (wrong color)", (10, 22), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                else:
                    cv2.putText(vis, "No circles detected - adjust lighting/position", (10, 22), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # --- Apply current bounds and show mask ---
        mask = hsv_inrange_wrap(
            hsv,
            hsv_bounds["h_lo"], hsv_bounds["h_hi"],
            hsv_bounds["s_lo"], hsv_bounds["s_hi"],
            hsv_bounds["v_lo"], hsv_bounds["v_hi"]
        )

        # Clean mask slightly
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

        # Overlay bounds
        txt = f"H[{hsv_bounds['h_lo']},{hsv_bounds['h_hi']}] S[{hsv_bounds['s_lo']},{hsv_bounds['s_hi']}] V[{hsv_bounds['v_lo']},{hsv_bounds['v_hi']}]"
        txt_color = (0, 255, 0) if updated_this_frame else (255, 255, 255)
        cv2.putText(vis, txt, (10, FRAME_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2)
        
        # Add instruction text
        instruct_txt = "Press SPACE to lock/unlock, Q to quit"
        cv2.putText(vis, instruct_txt, (10, FRAME_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        cv2.imshow("Ball Detection (camera)", vis)
        cv2.imshow("Ball Mask", mask)
        cv2.imshow("Ball Only", masked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space key to lock/unlock
            locked = not locked
            ball_confirmed = False  # Reset confirmation when toggling lock
            if locked:
                print("✓ Detection LOCKED - learning from detected ball")
            else:
                print("✗ Detection UNLOCKED")
        elif key == ord('t'):  # Toggle debug mode
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('b'):  # Toggle ROI box display
            show_roi = not show_roi
            print(f"ROI box: {'ON' if show_roi else 'OFF'}")
        elif key == ord('a') or key == 65:  # 'a' or Left arrow - decrease ROI size
            roi_fraction = max(0.3, roi_fraction - 0.05)
            print(f"ROI size: {int(roi_fraction*100)}% of frame")
        elif key == ord('e') or key == 68:  # 'e' or Right arrow - increase ROI size
            roi_fraction = min(1.0, roi_fraction + 0.05)
            print(f"ROI size: {int(roi_fraction*100)}% of frame")
        elif key == ord('r'):  # Show rejection reasons
            print(f"\n--- Rejection Reasons ({len(last_rejection_reasons)} circles rejected) ---")
            if last_rejection_reasons:
                for reason in last_rejection_reasons:
                    print(f"  {reason}")
            else:
                print("  No circles rejected (all passed filters)")
            print("---\n")
        elif key == 82 or key == ord('w'):  # Up arrow or 'w' - increase PARAM2
            hough_param2 += 1
            print(f"HOUGH_PARAM2 = {hough_param2}")
        elif key == 84 or key == ord('s'):  # Down arrow or 's' - decrease PARAM2
            hough_param2 = max(1, hough_param2 - 1)
            print(f"HOUGH_PARAM2 = {hough_param2}")
        elif key == ord('+') or key == ord('='):  # Increase PARAM1
            hough_param1 += 5
            print(f"HOUGH_PARAM1 = {hough_param1}")
        elif key == ord('-') or key == ord('_'):  # Decrease PARAM1
            hough_param1 = max(10, hough_param1 - 5)
            print(f"HOUGH_PARAM1 = {hough_param1}")

    picam2.stop()
    cv2.destroyAllWindows()

    print("\n" + "="*70)
    if ball_confirmed:
        print("✓ Ball detection successful!")
        print("Final HSV bounds learned from tennis ball (paste into your script):")
    else:
        print("✗ No confirmed ball detection.")
        print("Bounds may not be accurate. Point camera at a tennis ball and:")
        print("  1. Wait for green circle (good detection)")
        print("  2. Press SPACE to lock")
        print("  3. Let it learn for a few seconds")
        print("  4. Press Q to save")
        print("\nLast HSV bounds:")
    
    print(f"HSV_LOWER = np.array([{hsv_bounds['h_lo']}, {hsv_bounds['s_lo']}, {hsv_bounds['v_lo']}], dtype=np.uint8)")
    print(f"HSV_UPPER = np.array([{hsv_bounds['h_hi']}, {hsv_bounds['s_hi']}, {hsv_bounds['v_hi']}], dtype=np.uint8)")
    print("="*70)


if __name__ == "__main__":
    main()