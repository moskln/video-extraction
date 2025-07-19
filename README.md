# Licensed under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
# Copyright (c) 2025 Khoulane Moussa
# You may share and adapt this code non-commercially with attribution.
# Full license: https://creativecommons.org/licenses/by-nc/4.0/



\# 🎥 Video Extraction and OCR Tool



This tool processes videos from URLs (e.g. YouTube), extracts frames, detects structured regions like text blocks and charts using YOLOv8, and performs OCR using EasyOCR. The result is structured and visualized for export (e.g. as JSON and PDF).



---



\## 📂 Project Structure



video-extraction/

│

├── app.py # Flask web app

├── generator.py # Main orchestrator for extraction

├── generator\_content.py # Logic to format structured outputs

├── processor.py # Core video and OCR pipeline

├── DejaVuSans.ttf # Font for visualization

├── DejaVuSans-Bold.ttf

│

├── static/

│ ├── ocr\_text.pdf # Output file (sample)

│ └── structured\_data.json

│

├── templates/

│ └── index.html # Web UI

│

├── structured\_extractor/ # Layout + OCR logic

│ ├── detector.py

│ ├── layout\_detector.py

│ ├── ocr.py

│ └── postprocess.py

│

├── temp\_crops/ # Intermediate cropped images

└── pycache/ # Python bytecode cache





---



\## 🚀 Features



\- 🔗 Download videos from URLs

\- 🖼️ Extract key frames

\- 📐 Detect layout regions (table, text block, etc.)

\- 📝 Extract multilingual OCR (EasyOCR: Arabic + English)

\- 🧹 Clean, wrap, and format text

\- 📦 Export structured JSON and PDF

\- 🖼️ Visual overlays with bounding boxes



---



\## 🛠️ Setup



1\. \*\*Install dependencies\*\*:

```bash

pip install -r requirements.txt



