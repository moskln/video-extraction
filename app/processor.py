# processor.py
import os
import tempfile
import json
import threading
import base64
import textwrap
import subprocess

import cv2
import pytesseract
from PIL import Image
from fpdf import FPDF
import numpy as np
import whisper
import yt_dlp

from app.logger import setup_logger
from app.generator import generate_course_content
from ai_modules.detector import detect_structured_data
from ai_modules.explainer import explain_structured_data
from ai_modules.visual_report import generate_visual_report
from skimage.metrics import structural_similarity as ssim 

logger = setup_logger()


# 🧼 IMAGE PREPROCESSING
def preprocess_image(pil_img):
    open_cv_img = np.array(pil_img.convert('RGB'))
    gray_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGB2GRAY)
    _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh_img)


def ocr_image(image_path, lang_code="eng"):
    try:
        img = Image.open(image_path)
        processed_img = preprocess_image(img)
        return pytesseract.image_to_string(processed_img, lang=lang_code).strip()
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return ""


def is_image_blank(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img is None or np.mean(img) > 240


# 🎷 AUDIO TRANSCRIPTION
def get_audio_duration(filename):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return float(json.loads(result.stdout)['format']['duration'])
    except Exception as e:
        logger.error(f"Could not get audio duration: {e}")
        return 0


def transcribe_youtube_video(url, language_code=None, tmpdir=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'quiet': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = os.path.join(tmpdir, "audio.mp3")
    duration_minutes = get_audio_duration(audio_path) / 60

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="transcribe", language=language_code)

    return {"transcript": result["text"], "duration_minutes": duration_minutes}


# 🗄️ FRAME EXTRACTION
def extract_frames_from_video(video_path, temp_dir, frame_interval_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_to_skip = int(fps * frame_interval_sec)

    images = []
    count = saved_count = 0
    last_frame_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frames_to_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if last_frame_gray is not None:
                sim = ssim(gray, last_frame_gray)
                if sim > 0.95:
                    logger.info(f" Skipping frame {count} (SSIM {sim:.2f})")
                    count += 1
                    continue

            last_frame_gray = gray
            img_path = os.path.join(temp_dir, f"frame_{saved_count}.png")
            cv2.imwrite(img_path, frame)

            if not is_image_blank(img_path):
                images.append(img_path)
            else:
                os.remove(img_path)

            saved_count += 1

        count += 1

    cap.release()
    return images


# 🧠 OCR Multithreaded
def ocr_image_async(image_path, results, index, lang_code="eng"):
    results[index] = ocr_image(image_path, lang_code)


# 📦 EXPORT HELPERS
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_ocr_text_to_pdf(ocr_texts, output_path="static/ocr_text.pdf"):
    if not ocr_texts:
        logger.info("No OCR text to write.")
        return

    combined_text = "\n\n".join(ocr_texts)
    wrapped_text = "\n".join(textwrap.fill(line, width=100) for line in combined_text.split("\n"))

    pdf = FPDF()
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", "", 12)

    pdf.multi_cell(0, 10, wrapped_text)
    pdf.output(output_path)
    logger.info(f"OCR text PDF saved to: {output_path}")


# 🔄 MAIN PROCESSING PIPELINE
def combined_process(video_url, language="en"):
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Processing video: {video_url}")

        language_code = "ar" if language == "ar" else None
        transcript_data = transcribe_youtube_video(video_url, language_code, tmpdir)
        transcript = transcript_data["transcript"]
        duration = transcript_data["duration_minutes"]

        video_path = os.path.join(tmpdir, "video.mp4")
        with yt_dlp.YoutubeDL({
            'format': 'best',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True
        }) as ydl:
            ydl.download([video_url])

        frame_interval = max(1, min(10, duration / 20))
        images = extract_frames_from_video(video_path, tmpdir, frame_interval)

        ocr_results = [None] * len(images)
        ocr_lang = "ara" if language == "ar" else "eng"
        threads = [
            threading.Thread(target=ocr_image_async, args=(img, ocr_results, i, ocr_lang))
            for i, img in enumerate(images)
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]

        structured_outputs = []
        explanations = []
        tags_per_frame = []

        for img_path in images:
            try:
                structured = detect_structured_data(img_path)
                structured_outputs.append(structured)

                if not isinstance(structured, dict):
                     raise TypeError(f"Expected dict, got {type(structured)}")
 
                explanation = explain_structured_data(structured)
                tags = list(structured.get("grouped_wrapped_texts", {}).keys())
            except Exception as e:
                logger.error(f"Error processing frame {img_path}: {e}")
                explanation = "No explanation available."
                tags = []

            explanations.append(explanation)
            tags_per_frame.append({"frame": os.path.basename(img_path), "tags": tags})

        os.makedirs("static", exist_ok=True)
        with open("static/frame_explanations.txt", "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(explanations))

        with open("static/structured_data.json", "w", encoding="utf-8") as f:
            json.dump(structured_outputs, f, ensure_ascii=False, indent=2)

        generate_visual_report(images, explanations, output_pdf="static/visual_report.pdf")
        ocr_texts = [txt for txt in ocr_results if txt and txt.strip()]
        save_ocr_text_to_pdf(ocr_texts, output_path="static/ocr_text.pdf")

        combined_text = f"{transcript.strip()}\n\n" + "\n\n".join(txt.strip() for txt in ocr_texts if isinstance(txt, str))

        try:
            course_data = generate_course_content(combined_text, language, video_duration_minutes=duration)
            if not isinstance(course_data, dict):
                raise TypeError("generate_course_content() must return a dictionary")
        except Exception as e:
            logger.error(f"generate_course_content failed: {e}")
            raise

        course_data.update({
            "extracted_images_base64": [image_to_base64(img) for img in images],
            "frame_explanations": explanations,
            "frame_tags": tags_per_frame,
            "visual_report_pdf": "static/visual_report.pdf"
        })

        return course_data


# 🔁 Public interface for Celery/Flask
def process_video(video_url, language="en"):
    try:
        result = combined_process(video_url, language)
        return {
            "status": "success",
            "video_url": video_url,
            "frames_processed": len(result.get("extracted_images_base64", [])),
            "data": result
        }
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
