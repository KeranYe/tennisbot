from ultralytics import YOLO
from pathlib import Path
import shutil
import argparse

# ---- user settings ----
weights = "yolo26n.pt"               # or yolov8n.pt, etc.
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=".", help="Output directory for exported NCNN files")
parser.add_argument("--imgsz", type=int, default=416, help="Export image size. Smaller is usually faster.")
parser.add_argument("--batch", type=int, default=1, help="Export batch size (use 1 for real-time camera).")
parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization for faster inference.")
parser.add_argument("--data", type=str, default=None, help="Dataset YAML path for INT8 calibration (e.g., coco8.yaml).")
parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of calibration dataset to use for INT8.")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- export ----
model = YOLO(weights)
export_kwargs = {
    "format": "ncnn",
    "imgsz": args.imgsz,
    "batch": args.batch,
}

if args.int8:
    print("[WARN] --int8 is not supported by Ultralytics NCNN export in this version. Exporting without INT8.")
    if args.data or args.fraction != 1.0:
        print("[WARN] --data/--fraction are ignored unless INT8 export is supported.")

exported = model.export(**export_kwargs)  # creates *_ncnn_model directory by default

# ---- move to your target path ----
# exported is typically a pathlib.Path or a string path depending on version
export_path = Path(str(exported))
export_name = export_path.name
if export_name.endswith("_ncnn_model"):
    export_name = export_name.replace("_ncnn_model", f"_{args.imgsz}_ncnn_model")
else:
    export_name = f"{export_name}_{args.imgsz}"

dst = out_dir / export_name

if dst.exists():
    if dst.is_dir():
        shutil.rmtree(dst)
    else:
        dst.unlink()

shutil.move(str(export_path), str(dst))

weights_path = Path(weights)
if weights_path.exists():
    weights_path.unlink()

print("Exported NCNN saved to:", dst.resolve())
print(
    "Export options:",
    {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "int8": False,
        "data": args.data,
        "fraction": args.fraction,
    },
)
