from ultralytics import YOLO
from pathlib import Path
import shutil
import argparse

# ---- user settings ----
weights = "yolo26n.pt"               # or yolov8n.pt, etc.
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=".", help="Output directory for exported ONNX file")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- export ----
model = YOLO(weights)
exported = model.export(format="onnx")  # creates yolo26n.onnx next to weights (by default)

# ---- move to your target path ----
# exported is typically a pathlib.Path or a string path depending on version
export_path = Path(str(exported))
dst = out_dir / export_path.name
shutil.move(str(export_path), str(dst))

pt_path = export_path.with_suffix(".pt")
if pt_path.exists():
	pt_path.unlink()

print("Exported ONNX saved to:", dst.resolve())