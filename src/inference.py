import cv2
import glob
import numpy as np
import os
import random
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_PT = os.path.join(BASE_DIR, "weights", "primary.pt")
PLATE_PT   = os.path.join(BASE_DIR, "weights", "plate.pt")
FACE_PT    = os.path.join(BASE_DIR, "weights", "face.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CROP_SIZE  = 256
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Models ───────────────────────────────────────────────────────────────
primary   = YOLO(PRIMARY_PT)
plate_det = YOLO(PLATE_PT)
face_det  = YOLO(FACE_PT)

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(image_path, save_path="output.jpg", crop_size=CROP_SIZE):
    img     = cv2.imread(image_path)
    results = primary(img)[0]

    for box in results.boxes:
        cls             = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi             = img[y1:y2, x1:x2].copy()
        label           = "car" if cls == 0 else "person"
        box_color       = (0, 0, 255) if cls == 0 else (255, 165, 0)

        if cls == 0:  # car → detect plate
            for i, pb in enumerate(plate_det(roi)[0].boxes):
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                cv2.rectangle(roi, (px1, py1), (px2, py2), (0, 255, 0), 2)
                plate_crop = roi[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    new_w         = crop_size * 2
                    new_h         = max(int(plate_crop.shape[0] * new_w / plate_crop.shape[1]), 64)
                    plate_resized = cv2.resize(plate_crop, (new_w, new_h),
                                               interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"plate_crop_{i}.jpg"), plate_resized)

        elif cls == 1:  # person → detect face
            for i, fb in enumerate(face_det(roi)[0].boxes):
                fx1, fy1, fx2, fy2 = map(int, fb.xyxy[0])
                cv2.rectangle(roi, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                face_crop = roi[fy1:fy2, fx1:fx2]
                if face_crop.size > 0:
                    h, w     = face_crop.shape[:2]
                    scale    = crop_size / max(h, w)
                    new_w    = int(w * scale)
                    new_h    = int(h * scale)
                    resized  = cv2.resize(face_crop, (new_w, new_h),
                                          interpolation=cv2.INTER_CUBIC)
                    canvas   = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    pad_top  = (crop_size - new_h) // 2
                    pad_left = (crop_size - new_w) // 2
                    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"face_crop_{i}.jpg"), canvas)

        img[y1:y2, x1:x2] = roi
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")

    # Print detected crops
    plates = sorted(glob.glob(os.path.join(OUTPUT_DIR, "plate_crop_*.jpg")))
    faces  = sorted(glob.glob(os.path.join(OUTPUT_DIR, "face_crop_*.jpg")))
    if plates: print(f"🟩 {len(plates)} plate(s) saved: {plates}")
    if faces:  print(f"🟦 {len(faces)} face(s) saved: {faces}")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vehicle & People Surveillance Pipeline")
    parser.add_argument("--image",  type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output")
    parser.add_argument("--crop-size", type=int, default=256, help="Face crop thumbnail size")
    args = parser.parse_args()

    run_pipeline(args.image, save_path=args.output, crop_size=args.crop_size)

