import time
import cv2
import numpy as np
import sys
from picamera2 import Picamera2

FRAME_W, FRAME_H = 640, 480
CAMERA_NUM = 1  # Set to 0 or 1 to select CSI camera

def nothing(x):
    pass

def main():
    # Allow camera number to be overridden via command-line argument
    camera_num = CAMERA_NUM
    if len(sys.argv) > 1:
        try:
            camera_num = int(sys.argv[1])
            print(f"Using camera CSI {camera_num}")
        except ValueError:
            print(f"Invalid camera number: {sys.argv[1]}")
            sys.exit(1)

    picam2 = Picamera2(camera_num=camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
        buffer_count=4
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 420, 260)

    # Trackbars: start broad
    cv2.createTrackbar("H_low",  "Controls", 30, 179, nothing)
    cv2.createTrackbar("H_high", "Controls", 130, 179, nothing)
    cv2.createTrackbar("S_low",  "Controls", 40, 255, nothing)
    cv2.createTrackbar("S_high", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V_low",  "Controls", 40, 255, nothing)
    cv2.createTrackbar("V_high", "Controls", 255, 255, nothing)

    print("Click on the image to inspect HSV values. Press q to quit.")

    clicked_hsv = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_hsv, last_hsv
        if event == cv2.EVENT_LBUTTONDOWN:
            if last_hsv is not None and 0 <= y < last_hsv.shape[0] and 0 <= x < last_hsv.shape[1]:
                clicked_hsv = last_hsv[y, x].copy()
                print(f"Clicked pixel ({x},{y}) HSV = {clicked_hsv}")

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_mouse)

    last_hsv = None

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        last_hsv = hsv

        h_low  = cv2.getTrackbarPos("H_low", "Controls")
        h_high = cv2.getTrackbarPos("H_high", "Controls")
        s_low  = cv2.getTrackbarPos("S_low", "Controls")
        s_high = cv2.getTrackbarPos("S_high", "Controls")
        v_low  = cv2.getTrackbarPos("V_low", "Controls")
        v_high = cv2.getTrackbarPos("V_high", "Controls")

        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        # Handle hue wrap-around if H_low > H_high
        if h_low <= h_high:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            lower1 = np.array([0, s_low, v_low], dtype=np.uint8)
            upper1 = np.array([h_high, s_high, v_high], dtype=np.uint8)
            lower2 = np.array([h_low, s_low, v_low], dtype=np.uint8)
            upper2 = np.array([179, s_high, v_high], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                                  cv2.inRange(hsv, lower2, upper2))

        # Clean mask a bit
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, k5, iterations=2)

        result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_clean)

        overlay = frame_bgr.copy()
        txt = f"H[{h_low},{h_high}] S[{s_low},{s_high}] V[{v_low},{v_high}]"
        cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if clicked_hsv is not None:
            cv2.putText(overlay, f"Clicked HSV: {clicked_hsv.tolist()}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Camera", overlay)
        cv2.imshow("Mask", mask_clean)
        cv2.imshow("Masked Result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nUse these values in your script:")
            print(f"HSV_LOWER = np.array([{h_low}, {s_low}, {v_low}], dtype=np.uint8)")
            print(f"HSV_UPPER = np.array([{h_high}, {s_high}, {v_high}], dtype=np.uint8)")
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()