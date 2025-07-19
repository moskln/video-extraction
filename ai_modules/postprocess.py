# structured_extractor/postprocess.py

import re
import textwrap

def clean_ocr_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF€$%:.,\-–—+=()/\\[\]{}|<>!?@#&*\'\"]', '', text)
    return text.strip()

def wrap_cleaned_text(text, width=100):
    return "\n".join(
        textwrap.fill(line, width=width)
        for line in text.split("\n")
        if line.strip()
    )
