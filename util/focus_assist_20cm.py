#!/usr/bin/env python3
import time
import sys
import cv2
import numpy as np
from picamera2 import Picamera2

# Try importing lens controls (only available/supported on some cameras)
# try:
    # from libcamera import controls
    # HAS_LIBCAMERA_CONTROLS = True
# except Exception:
    # HAS_LIBCAMERA_CONTROLS = False

HAS_LIBCAMERA_CONTROLS = False


def variance_of_laplacian(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main(camera_num: int = 0):
    picam2 = Picamera2(camera_num=camera_num)

    # 480p (640x480), RGB for easy OpenCV processing
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)

    print(f"Using camera: CSI {camera_num}")
    print("Camera model:", picam2.camera_properties.get("Model", "Unknown"))
    print("\nAvailable camera controls (keys):")
    print(sorted(list(picam2.camera_controls.keys())))

    picam2.start()
    time.sleep(1.0)

    # OPTIONAL: if camera supports software lens control, set ~20 cm focus
    # 20 cm = 0.20 m => LensPosition ~= 1/0.20 = 5.0 dioptres
    if HAS_LIBCAMERA_CONTROLS and "LensPosition" in picam2.camera_controls and "AfMode" in picam2.camera_controls:
        try:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 5.0})
            print("\nSet software focus to approx 20 cm (LensPosition=5.0).")
            print("Note: lens calibration is approximate; verify by sharpness score.")
            time.sleep(0.5)
        except Exception as e:
            print("\nCould not set LensPosition (camera may be manual-focus only):", e)
            print("Please adjust focus physically on the lens.")
    else:
        print("\nNo LensPosition/AfMode controls found.")
        print("This is expected on manual-focus cameras (e.g., official AI Camera IMX500).")
        print("Please adjust focus physically on the lens while watching the sharpness score.")

    print("\nInstructions:")
    print("- Place a chessboard / textured target at ~20 cm from the lens")
    print("- Adjust focus (physical lens ring if needed) to maximize sharpness")
    print("- Press 'q' to quit")

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        # center ROI (40% width x 40% height)
        rx0 = int(w * 0.30)
        ry0 = int(h * 0.30)
        rx1 = int(w * 0.70)
        ry1 = int(h * 0.70)

        roi = gray[ry0:ry1, rx0:rx1]
        sharpness = variance_of_laplacian(roi)

        # overlay
        cv2.rectangle(frame_bgr, (rx0, ry0), (rx1, ry1), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Sharpness (ROI): {sharpness:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame_bgr, "Target distance: ~20 cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_bgr, "Press q to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Focus Assist (640x480)", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_num = 1 # 0 or 1
    
    main(camera_num=camera_num)