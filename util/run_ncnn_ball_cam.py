#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO


def find_class_id(model: YOLO, class_name: str) -> int:
    """Find class index by name from Ultralytics model.names."""
    names = model.names
    if isinstance(names, dict):
        inv = {v: k for k, v in names.items()}
        if class_name not in inv:
            raise ValueError(f"Class '{class_name}' not found. Available: {list(inv.keys())[:20]} ...")
        return int(inv[class_name])
    else:
        if class_name not in names:
            raise ValueError(f"Class '{class_name}' not found. Available: {names[:20]} ...")
        return int(names.index(class_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Picamera2 camera index (CSI0 often 0).")
    ap.add_argument("--model", type=str, default="yolo26n_ncnn_model",
                    help="Path to exported NCNN model folder (e.g., yolo26n_ncnn_model).")
    ap.add_argument("--class_name", type=str, default="sports ball",
                    help="Target class name. COCO pretrained uses 'sports ball'.")
    ap.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    ap.add_argument("--height", type=int, default=720, help="Camera capture height.")
    ap.add_argument("--fps", type=int, default=30, help="Camera FPS.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (smaller is faster).")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    ap.add_argument("--max_det", type=int, default=10, help="Max detections per frame.")
    ap.add_argument("--display", action="store_true", help="Show annotated window.")
    ap.add_argument("--print_each", action="store_true", help="Print detections every frame.")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"NCNN model path not found: {model_path.resolve()}")

    # Load NCNN model exported by Ultralytics (folder)
    model = YOLO(str(model_path))

    # Find target class id
    target_cls = find_class_id(model, args.class_name)
    print(f"[INFO] Target class '{args.class_name}' id={target_cls}")

    # Setup CSI camera via Picamera2
    picam2 = Picamera2(camera_num=args.camera)
    cfg = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": args.fps},
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.5)

    print("[INFO] Running. Ctrl+C to stop. Press 'q' in window to quit (if --display).")
    try:
        while True:
            rgb = picam2.capture_array("main")  # RGB888 numpy array

            t0 = time.time()
            # Run inference (CPU). With NCNN export, this uses the NCNN backend internally.
            res = model.predict(
                source=rgb,
                imgsz=args.imgsz,
                conf=args.conf,
                max_det=args.max_det,
                verbose=False,
                device="cpu",
            )[0]
            dt_ms = (time.time() - t0) * 1000.0

            det_list = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    if cls != target_cls:
                        continue
                    conf = float(boxes.conf[i].item())
                    x0, y0, x1, y1 = [float(v) for v in boxes.xyxy[i].tolist()]
                    det_list.append((conf, x0, y0, x1, y1))

            det_list.sort(key=lambda x: x[0], reverse=True)

            if args.print_each:
                if det_list:
                    best = det_list[0]
                    print(f"[DET] n={len(det_list)} best_conf={best[0]:.2f}  infer={dt_ms:.1f}ms")
                else:
                    print(f"[DET] n=0  infer={dt_ms:.1f}ms")

            if args.display:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"infer {dt_ms:.1f} ms", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                for conf, x0, y0, x1, y1 in det_list:
                    cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                    cv2.putText(bgr, f"{args.class_name} {conf:.2f}",
                                (int(x0), max(20, int(y0) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("NCNN YOLO Tennis Ball Detection", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()