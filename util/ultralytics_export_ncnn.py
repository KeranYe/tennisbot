from ultralytics import YOLO
from pathlib import Path
import shutil
import argparse

# ---- user settings ----
weights = "yolo26n.pt"               # or yolov8n.pt, etc.
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=".", help="Output directory for exported NCNN files")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- export ----
model = YOLO(weights)
exported = model.export(format="ncnn")  # creates *_ncnn_model directory by default

# ---- move to your target path ----
# exported is typically a pathlib.Path or a string path depending on version
export_path = Path(str(exported))
dst = out_dir / export_path.name

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
