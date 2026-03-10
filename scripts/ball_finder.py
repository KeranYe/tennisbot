#!/usr/bin/env python3
import time
import math
import argparse
import cv2
import numpy as np

from picamera2 import Picamera2, MappedArray

# IMX500 imports (paths vary slightly across installs)
try:
    from picamera2.devices import IMX500
except Exception:
    from picamera2.devices.imx500 import IMX500

from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# =========================
# User variables (defaults)
# =========================
CALIB_FILE = "../data/camera_intrinsics_640x480_20cmfocus.npz"
FRAME_W, FRAME_H = 640, 480  # MUST match calibration resolution (or scale intrinsics)

# Geometry
CAMERA_HEIGHT_M = 0.306            # <-- you said ~30.6 cm above ground
CAMERA_PITCH_DEG = 90.0             # pitch DOWN from horizontal: 0=horizontal, 90=straight down
BALL_DIAMETER_M = 0.067             # tennis ball
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0

# Detector model (adjust if you use a different .rpk)
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# Detection selection
TARGET_LABELS = {"sports ball"}     # COCO label for tennis ball in many models
CONF_MIN = 0.10

# Allow detections anywhere in frame (no center preference)
USE_CENTER_GATE = False             # Set to True to restrict to center region only
CENTER_GATE_PX = 320                # max pixel distance from image center (only if USE_CENTER_GATE=True)
CENTER_WEIGHT = 0.0                 # no center preference in scoring (0.0 = pure confidence)

# Smooth output (EMA)
EMA_ALPHA = 0.25

# =========================
# Helpers
# =========================
def load_calibration(npz_path: str):
    data = np.load(npz_path)
    K = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)
    return K, dist

def undistort_pixel_to_ray(u, v, K, dist):
    """
    Pixel (u,v) -> normalized ray r=[x,y,1] in camera frame (OpenCV convention).
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=None)
    x = und[0, 0, 0]
    y = und[0, 0, 1]
    ray = np.array([x, y, 1.0], dtype=np.float64)
    ray_unit = ray / np.linalg.norm(ray)
    return ray, ray_unit

def intersect_ray_with_ball_center_plane(ray, camera_height_m, pitch_down_deg, ball_radius_m):
    """
    Plane: y*cos(theta) + z*sin(theta) = H - R
    where theta = pitch_down_deg (0=horizontal, 90=straight down)
    """
    H = float(camera_height_m)
    R = float(ball_radius_m)
    if H <= R:
        return None

    theta = math.radians(float(pitch_down_deg))
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    rhs = H - R

    denom = float(np.dot(n, ray))
    if denom <= 1e-8:
        return None
    t = rhs / denom
    if t <= 0:
        return None
    return t * ray

def default_coco_labels():
    # Fallback COCO labels list (includes "-" placeholders) commonly used in examples
    return [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","-","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","-","backpack","umbrella","-","-","handbag","tie","suitcase",
        "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","bottle","-","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
        "chair","couch","potted plant","bed","-","dining table","-","-","toilet","-","tv","laptop",
        "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","-",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    ]

def fmt_cm(p):
    if p is None:
        return "None"
    return f"[{p[0]*100: .1f}, {p[1]*100: .1f}, {p[2]*100: .1f}] cm"


class BallFinder:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        frame_w=FRAME_W,
        frame_h=FRAME_H,
        camera_height_m=CAMERA_HEIGHT_M,
        camera_pitch_deg=CAMERA_PITCH_DEG,
        ball_radius_m=BALL_RADIUS_M,
        ema_alpha=EMA_ALPHA,
        center_gate_px=CENTER_GATE_PX,
        center_weight=CENTER_WEIGHT,
        use_center_gate=USE_CENTER_GATE,
        aspect_threshold=1.3,
        edge_margin_px=50,
        inlet_block_y_frac=0.55,
    ):
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.camera_height_m = float(camera_height_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.ball_radius_m = float(ball_radius_m)
        self.ema_alpha = float(ema_alpha)
        self.center_gate_px = float(center_gate_px)
        self.center_weight = float(center_weight)
        self.use_center_gate = bool(use_center_gate)
        self.aspect_threshold = float(aspect_threshold)
        self.edge_margin_px = int(edge_margin_px)
        self.inlet_block_y_frac = float(inlet_block_y_frac)
        self.cx0 = self.frame_w / 2.0
        self.cy0 = self.frame_h / 2.0
        self.ema_p = None

    def compute_ball_center(self, x, y, w, h):
        aspect_ratio = w / max(h, 1.0)

        u_default = x + 0.5 * w
        v_default = y + 0.5 * h

        if aspect_ratio > self.aspect_threshold:
            u = u_default
            estimated_radius = w / 2.0

            # Wide bbox means vertical clipping/occlusion.
            # In this setup, inlet is fixed in lower half and can block the BALL BOTTOM
            # when the ball is near the inlet area.
            #
            # - bottom occluded: top edge is reliable  -> center_y = top + r
            # - top clipped:     bottom edge reliable -> center_y = bottom - r
            inlet_y = self.inlet_block_y_frac * self.frame_h
            near_inlet = (y + h) >= inlet_y
            if near_inlet:
                v = y + estimated_radius
            else:
                v = (y + h) - estimated_radius
        elif aspect_ratio < (1.0 / self.aspect_threshold):
            v = v_default
            estimated_radius = h / 2.0
            bbox_center_x = x + w / 2.0
            if x < self.edge_margin_px:
                u = (x + w) - estimated_radius
            elif (x + w) > (self.frame_w - self.edge_margin_px):
                u = x + estimated_radius
            else:
                if bbox_center_x < self.frame_w / 2:
                    u = (x + w) - estimated_radius
                else:
                    u = x + estimated_radius
        else:
            u = u_default
            v = v_default

        u = float(np.clip(u, 0.0, self.frame_w - 1.0))
        v = float(np.clip(v, 0.0, self.frame_h - 1.0))
        return u, v

    def detection_score(self, conf, u, v):
        d_center = math.hypot(u - self.cx0, v - self.cy0)
        if self.use_center_gate and d_center > self.center_gate_px:
            return None, d_center
        center_score = 1.0 - min(1.0, d_center / max(1.0, self.center_gate_px))
        score_total = (1.0 - self.center_weight) * conf + self.center_weight * center_score
        return score_total, d_center

    def project_to_3d(self, u, v):
        ray, _ = undistort_pixel_to_ray(u, v, self.K, self.dist)
        p_cam = intersect_ray_with_ball_center_plane(
            ray,
            self.camera_height_m,
            self.camera_pitch_deg,
            self.ball_radius_m,
        )
        if p_cam is not None:
            if self.ema_p is None:
                self.ema_p = p_cam.copy()
            else:
                self.ema_p = (1 - self.ema_alpha) * self.ema_p + self.ema_alpha * p_cam
        return p_cam, self.ema_p

    @staticmethod
    def camera_to_robot_frame(p_cam):
        """
        Convert camera frame (x right, y down, z forward)
        to robot body frame (x forward, y left, z up).
        Returns array ordered as [x, y, z] = [forward, left, up].
        """
        if p_cam is None:
            return None
        # Robot frame: x=forward, y=left, z=up
        # x_robot = p_cam[2]      # forward (from camera z)
        # y_robot = -p_cam[0]     # left (from camera -x)
        # z_robot = -p_cam[1]     # up (from camera -y)
        
        x_robot = -p_cam[1]
        y_robot = -p_cam[0]
        z_robot = -p_cam[2]

        # Return in order: [x, y, z] for proper index order
        return np.array([x_robot, y_robot, z_robot], dtype=np.float64)

    def clip_status(self, x, y, w, h):
        aspect_ratio = w / max(h, 1.0)
        if aspect_ratio > self.aspect_threshold:
            inlet_y = self.inlet_block_y_frac * self.frame_h
            near_inlet = (y + h) >= inlet_y
            if near_inlet:
                return aspect_ratio, (0, 140, 255), " [BOTTOM-occluded@inlet]"
            return aspect_ratio, (0, 165, 255), " [TOP-clip]"
        if aspect_ratio < (1.0 / self.aspect_threshold):
            if x < self.edge_margin_px:
                side = "LEFT"
            elif (x + w) > (self.frame_w - self.edge_margin_px):
                side = "RIGHT"
            else:
                side = "SIDE"
            return aspect_ratio, (255, 165, 0), f" [{side}-clip]"
        return aspect_ratio, (0, 255, 0), ""

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IMX500 ball localization with 3D position estimation")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        choices=[0, 1],
        help="Camera source: 0 (CSI 0) or 1 (CSI 1) (default: 0)"
    )
    args = parser.parse_args()

    K, dist = load_calibration(CALIB_FILE)
    print("Loaded calibration:", CALIB_FILE)
    print("Camera frame: x right, y down, z forward")
    print("Robot frame : x forward, y left, z up")
    print(f"H={CAMERA_HEIGHT_M:.3f} m, pitch_down={CAMERA_PITCH_DEG:.1f} deg, ball D={BALL_DIAMETER_M:.3f} m")

    # IMX500 must be created before Picamera2
    imx500 = IMX500(MODEL_PATH)
    intr = imx500.network_intrinsics
    if not intr:
        intr = NetworkIntrinsics()
        intr.task = "object detection"
    if intr.task != "object detection":
        raise RuntimeError("Loaded network is not object detection")

    intr.update_with_defaults()
    labels = intr.labels if intr.labels is not None else default_coco_labels()

    # Use specified camera number
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

    ball_finder = BallFinder(
        camera_matrix=K,
        dist_coeffs=dist,
        frame_w=FRAME_W,
        frame_h=FRAME_H,
        camera_height_m=CAMERA_HEIGHT_M,
        camera_pitch_deg=CAMERA_PITCH_DEG,
        ball_radius_m=BALL_RADIUS_M,
        ema_alpha=EMA_ALPHA,
        center_gate_px=CENTER_GATE_PX,
        center_weight=CENTER_WEIGHT,
        use_center_gate=USE_CENTER_GATE,
    )

    print("Running. Press 'q' to quit.")

    while True:
        request = picam2.capture_request()
        md = request.get_metadata()

        # Extract the RGB frame
        with MappedArray(request, "main") as m:
            frame_rgb = m.array.copy()

        # Get detection outputs
        dets = []
        np_outputs = imx500.get_outputs(md, add_batch=True)
        if np_outputs is not None:
            input_w, input_h = imx500.get_input_size()

            if intr.postprocess == "nanodet":
                # For nanodet models
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
                dets.append((score_total, conf, label, x, y, w, h, u, v, d_center))

        request.release()

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        chosen = max(dets, key=lambda t: t[0]) if dets else None

        if chosen is not None:
            score_total, conf, label, x, y, w, h, u, v, d_center = chosen

            p_cam, ema_p = ball_finder.project_to_3d(u, v)
            p_robot = ball_finder.camera_to_robot_frame(p_cam)
            ema_robot = ball_finder.camera_to_robot_frame(ema_p)
            aspect_ratio, bbox_color, clip_status = ball_finder.clip_status(x, y, w, h)

            # Draw
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), bbox_color, 2)
            cv2.circle(frame_bgr, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
            cv2.putText(frame_bgr, f"{label} conf={conf:.2f}{clip_status}", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

            cv2.putText(frame_bgr, f"3D (robot): {fmt_cm(p_robot)}", (10, FRAME_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_bgr, f"3D smooth : {fmt_cm(ema_robot)}", (10, FRAME_H - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show bbox aspect ratio for debugging
            cv2.putText(frame_bgr, f"BBox: {w}x{h} (AR={aspect_ratio:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        else:
            cv2.putText(frame_bgr, "No sports ball detection (try lowering CONF_MIN).",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame_bgr,
                    f"H={CAMERA_HEIGHT_M:.2f}m pitch_down={CAMERA_PITCH_DEG:.1f}deg D={BALL_DIAMETER_M:.3f}m",
                    (10, FRAME_H - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

        cv2.imshow("IMX500 bbox center -> 3D ball (camera frame)", frame_bgr)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()