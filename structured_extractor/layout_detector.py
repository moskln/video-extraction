# structured_extractor/layout_detector.py

from ultralytics import YOLO
import cv2
import os

# Load pre-trained YOLOv8 model
# If you have a custom model trained for layouts, replace with your .pt path
yolo_model = YOLO("yolov8x.pt")

# Example class mapping (update if using a custom-trained model)
DEFAULT_CLASSES = {
    0: "object",
    1: "chart",
    2: "table",
    3: "window",
    4: "text_block",
    5: "diagram"
}


def detect_layout_regions(image_path, class_filter=None, conf_threshold=0.4):
    if not os.path.exists(image_path):
        return []

    results = yolo_model(image_path, conf=conf_threshold)
    detected_regions = []

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = DEFAULT_CLASSES.get(cls_id, f"class_{cls_id}")

        if class_filter and label not in class_filter:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]

        detected_regions.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return detected_regions

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
