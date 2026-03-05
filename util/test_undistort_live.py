#!/usr/bin/env python3
import time
import sys
import cv2
import numpy as np
from picamera2 import Picamera2

CALIB_FILE = "../data/camera_intrinsics_640x480_20cmfocus.npz"
FRAME_SIZE = (640, 480)

def main(camera_num: int = 0):
    data = np.load(CALIB_FILE)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]

    picam2 = Picamera2(camera_num=camera_num)
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    print(f"Using camera: CSI {camera_num}")

    w, h = FRAME_SIZE
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

    print("Press q to quit")

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        undist = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

        # draw guide lines to visually assess straightness
        for y in [int(h*0.25), int(h*0.5), int(h*0.75)]:
            cv2.line(frame_bgr, (0, y), (w-1, y), (0, 255, 255), 1)
            cv2.line(undist,    (0, y), (w-1, y), (0, 255, 255), 1)

        combined = np.hstack([frame_bgr, undist])
        cv2.putText(combined, "Raw", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, "Undistorted", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Raw vs Undistorted", combined)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
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