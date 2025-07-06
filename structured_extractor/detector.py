# structured_extractor/detector.py

import os
import json
import cv2
from collections import defaultdict

from structured_extractor.ocr import perform_easyocr
from structured_extractor.postprocess import clean_ocr_text, wrap_cleaned_text
from structured_extractor.layout_detector import detect_layout_regions, crop_regions_from_image


def visualize_regions(image_path, regions, output_path=None):
    image = cv2.imread(image_path)
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        label = region["label"]
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if output_path is None:
        output_path = f"{os.path.splitext(image_path)[0]}_visualized.png"

    cv2.imwrite(output_path, image)
    print(f"[INFO] Saved visualization to: {output_path}")


def detect_structured_data(image_path):
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    # Step 1: Detect layout regions
    regions = detect_layout_regions(image_path, class_filter=["chart", "table", "window", "text_block"])
    cropped_regions = crop_regions_from_image(image_path, regions)

    # Sort regions top-down, left-right
    cropped_regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    grouped_texts = defaultdict(list)

    if cropped_regions:
        for region in cropped_regions:
            raw_text, language = perform_easyocr(region["crop_path"])
            cleaned = clean_ocr_text(raw_text)
            wrapped = wrap_cleaned_text(cleaned)

            region.update({
                "language": language,
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "wrapped_text": wrapped
            })

            grouped_texts[region["label"]].append({
                "text": wrapped,
                "bbox": region["bbox"]
            })

        # Save visual overlay
        visualize_regions(image_path, cropped_regions)

        result = {
            "image_path": image_path,
            "detected_regions": cropped_regions,
            "grouped_wrapped_texts": dict(grouped_texts)
        }

    else:
        # Fallback: OCR on the whole image
        print(f"[WARNING] No layout regions detected for: {image_path}")
        raw_text, language = perform_easyocr(image_path)
        cleaned = clean_ocr_text(raw_text)
        wrapped = wrap_cleaned_text(cleaned)

        result = {
            "image_path": image_path,
            "detected_regions": [],
            "language": language,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "grouped_wrapped_texts": {
                "full_image": [{"text": wrapped, "bbox": None}]
            }
        }

    # Save structured output to JSON
    json_path = f"{os.path.splitext(image_path)[0]}_extracted.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved JSON output to: {json_path}")

    return result
