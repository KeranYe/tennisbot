#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
from picamera2 import Picamera2

# # Optional AF/lens control support
# try:
#     from libcamera import controls
#     HAS_LIBCAMERA_CONTROLS = True
# except Exception:
#     HAS_LIBCAMERA_CONTROLS = False

HAS_LIBCAMERA_CONTROLS = False

# ========= USER SETTINGS =========
SAVE_DIR = "../data/calib_images_640x480"
FRAME_SIZE = (640, 480)  # 480p
CHESSBOARD_SIZE = (9, 6)  # internal corners (cols, rows)
USE_SOFTWARE_FOCUS = False  # Set True only if your camera supports LensPosition
LENS_POSITION_DIOPTRES = 5.0  # ~20 cm => 1/0.2 = 5.0
# =================================

def main(camera_num: int = 0):
    os.makedirs(SAVE_DIR, exist_ok=True)

    picam2 = Picamera2(camera_num=camera_num)
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)

    print(f"Using camera: CSI {camera_num}")
    print("Camera model:", picam2.camera_properties.get("Model", "Unknown"))
    print("Available controls:", sorted(list(picam2.camera_controls.keys())))

    picam2.start()
    time.sleep(1.0)

    # Optional software focus
    if USE_SOFTWARE_FOCUS:
        if HAS_LIBCAMERA_CONTROLS and "LensPosition" in picam2.camera_controls and "AfMode" in picam2.camera_controls:
            try:
                picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": LENS_POSITION_DIOPTRES})
                print(f"Set LensPosition={LENS_POSITION_DIOPTRES} (approx focus distance = {1.0/LENS_POSITION_DIOPTRES:.2f} m)")
                time.sleep(0.5)
            except Exception as e:
                print("Warning: Failed to set software focus:", e)
                print("Proceeding. If using official AI Camera, adjust focus physically.")
        else:
            print("LensPosition/AfMode not available on this camera. Adjust focus physically.")

    print("\nInstructions:")
    print("- Move chessboard around the frame (center, edges, corners)")
    print("- Tilt/rotate it; don't just keep it fronto-parallel")
    print("- Press 's' to save a frame when corners are detected")
    print("- Press 'q' to quit")
    print(f"- Saving images to: {SAVE_DIR}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    saved = 0

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame_bgr.copy()

        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners_refined, found)
            status_text = "Chessboard: FOUND (press 's' to save)"
            color = (0, 255, 0)
        else:
            status_text = "Chessboard: NOT FOUND"
            color = (0, 0, 255)

        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Saved: {saved}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, "q=quit, s=save", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        cv2.imshow("Calibration Capture (640x480)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            if found:
                filename = os.path.join(SAVE_DIR, f"img_{saved:03d}.png")
                cv2.imwrite(filename, frame_bgr)
                print("Saved:", filename)
                saved += 1
                time.sleep(0.15)  # slight debounce
            else:
                print("Pattern not found; not saved.")

    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {saved} images to {SAVE_DIR}")


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