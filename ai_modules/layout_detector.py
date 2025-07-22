# structured_extractor/layout_detector.py

import os
import time
import cv2
from ultralytics import YOLO

# Load YOLO model and move to GPU if available
model_path = os.path.join(os.path.dirname(__file__), '..', 'yolov8x.pt')
yolo_model = YOLO(model_path)
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    yolo_model.to("cuda")

DEFAULT_CLASSES = {
    0: "object",
    1: "chart",
    2: "table",
    3: "window",
    4: "text_block",
    5: "diagram"
}

CONFIDENCE_THRESHOLD = 0.4  # You can increase to e.g. 0.5 to be stricter


def detect_layout_regions(image_path, class_filter=None, conf_threshold=CONFIDENCE_THRESHOLD):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return [], [], []

    start_time = time.time()
    results = yolo_model(image_path, conf=conf_threshold)
    elapsed = time.time() - start_time

    detected_regions, texts, boxes = [], [], []
    skipped = 0

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        if conf < conf_threshold:
            skipped += 1
            continue

        label = DEFAULT_CLASSES.get(cls_id, f"class_{cls_id}")
        if class_filter and label not in class_filter:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        bbox = [x1, y1, x2, y2]

        detected_regions.append({
            "label": label,
            "confidence": conf,
            "bbox": bbox
        })

        boxes.append(bbox)
        texts.append(label)

    print(f"[INFO] Processed {image_path} in {elapsed:.2f}s with {len(detected_regions)} detections, skipped {skipped} low-conf boxes.")

    return detected_regions, texts, boxes


def crop_regions_from_image(image_path, regions, output_dir="temp_crops"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    crops = []

    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region["bbox"]
        cropped_img = image[y1:y2, x1:x2]
        crop_path = os.path.join(output_dir, f"crop_{i}_{region['label']}.png")
        cv2.imwrite(crop_path, cropped_img)
        region["crop_path"] = crop_path
        crops.append(region)

    return crops
