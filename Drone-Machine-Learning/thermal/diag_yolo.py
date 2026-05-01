"""
Run on Pi: python3 inference/diag_yolo.py --model inference/llvip_thermal_sar_best.pt
Saves diag_output.jpg with any detections drawn. No camera needed.
"""
import argparse, os, sys, glob
import cv2 as cv
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args = parser.parse_args()

print(f"Loading {args.model}...")
model = YOLO(args.model)
print("Model loaded OK")

# Test on actual saved diag frames
import glob
frames = sorted(glob.glob('/home/aurora/diag_frame_*.jpg'))
if not frames:
    print("No diag frames found at /home/aurora/diag_frame_*.jpg")
    sys.exit(1)

for path in frames:
    img = cv.imread(path)
    # Convert to grayscale and back to 3-channel to match inference pipeline
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    results = model(img_rgb, classes=[0], conf=0.01, verbose=False)
    boxes = results[0].boxes
    confs = [round(float(b.conf[0]), 3) for b in boxes]
    print(f"{path}: {len(boxes)} detections, confs={confs}")
    out = results[0].plot()
    out_path = path.replace('diag_frame_', 'diag_result_')
    cv.imwrite(out_path, out)
print("Done - check /home/aurora/diag_result_*.jpg")
