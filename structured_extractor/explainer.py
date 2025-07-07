import os
import json

def explain_structured_data(structured_json):
    grouped_texts = structured_json.get("grouped_wrapped_texts", {})
    detected_labels = list(grouped_texts.keys())

    explanations = []
    image_path = os.path.basename(structured_json.get("image_path", "frame"))

    explanations.append(f"🖼️ Explanation for frame: {image_path}")

    for label, regions in grouped_texts.items():
        if not regions:
            continue

        if label == "table":
            explanations.append(f"📊 A **table** is shown. It may contain structured data such as statistics or comparisons:")
        elif label == "chart":
            explanations.append(f"📈 A **chart** is present — possibly a bar, pie, or line chart visualizing trends or distributions:")
        elif label == "window":
            explanations.append(f"🪟 A **UI window** or web/app interface is visible — possibly showing user interactions or tools:")
        elif label == "text_block":
            explanations.append(f"📝 A **text block** is detected — may be part of a slide, subtitle, or annotation:")
        elif label == "diagram":
            explanations.append(f"📐 A **diagram or drawing** appears, which may explain a process or structure:")
        elif label == "object":
            explanations.append(f"👤 One or more **people or objects** are present in the frame — this may be real-world content or contextual visual:")
        elif label == "full_image":
            explanations.append(f"📷 The whole frame contains unstructured text or content:")

        for region in regions:
            text = region.get("text", "").strip()
            if text:
                explanations.append(f"→ {text}")

    if not explanations or len(explanations) == 1:
        explanations.append("⚠️ No visual elements detected in this frame.")

    return "\n".join(explanations)
