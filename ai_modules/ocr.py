# structured_extractor/ocr.py

import easyocr
from langdetect import detect
import cv2

def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in ["en", "ar"] else "en"
    except:
        return "en"

def perform_easyocr(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(["en", "ar"], gpu=False)
    results = reader.readtext(image)

    raw_text = " ".join([res[1] for res in results])
    language = detect_language(raw_text)

    return raw_text.strip(), language
