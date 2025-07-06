import os
import tempfile
import cv2
import pytesseract
from PIL import Image
import numpy as np
import whisper
import yt_dlp
import subprocess
import json
import threading
import base64
import io
import textwrap
from fpdf import FPDF
from generator import generate_course_content
from structured_extractor.detector import detect_structured_data

def preprocess_image(pil_img):
    # Convert to grayscale and binarize for better OCR
    open_cv_img = np.array(pil_img.convert('RGB'))
    gray_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGB2GRAY)
    _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh_img)

def ocr_image(image_path, lang_code="eng"):
    img = Image.open(image_path)
    processed_img = preprocess_image(img)
    text = pytesseract.image_to_string(processed_img, lang=lang_code)
    return text.strip()

def is_image_blank(image_path, threshold=10):
    # Simple check: if image brightness variance is too low, consider blank
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    mean_val = np.mean(img)
    return mean_val > 240  # Mostly white = blank

def get_audio_duration(filename):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", filename
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        ffprobe_output = json.loads(result.stdout)
        duration = float(ffprobe_output['format']['duration'])
        return duration
    except Exception as e:
        print(f"[ERROR] Could not get duration: {e}")
        return 0

def transcribe_youtube_video(url, language_code=None, tmpdir=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    filename = os.path.join(tmpdir, "audio.mp3")

    duration_seconds = get_audio_duration(filename)
    duration_minutes = duration_seconds / 60

    model = whisper.load_model("base")
    result = model.transcribe(filename, task="transcribe", language=language_code)

    return {
        "transcript": result["text"],
        "duration_minutes": duration_minutes
    }

def extract_frames_from_video(video_path, temp_dir, frame_interval_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback

    frames_to_skip = int(fps * frame_interval_sec)
    extracted_images = []
    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frames_to_skip == 0:
            img_path = os.path.join(temp_dir, f"frame_{saved_count}.png")
            cv2.imwrite(img_path, frame)
            if not is_image_blank(img_path):
                extracted_images.append(img_path)
            else:
                os.remove(img_path)  # Remove blank images early
            saved_count += 1
        count += 1
    cap.release()
    return extracted_images

def ocr_image_async(image_path, results, index, lang_code="eng"):
    try:
        text = ocr_image(image_path, lang_code)
        results[index] = text
    except Exception as e:
        results[index] = ""
        print(f"[ERROR] OCR failed for {image_path}: {e}")

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return b64_str



def save_ocr_text_to_pdf(ocr_texts, output_path="static/ocr_text.pdf"):
    if not ocr_texts:
        print("[INFO] No OCR text to write.")
        return

    combined_text = "\n\n".join(ocr_texts)
    wrapped_text = "\n".join(textwrap.fill(line, width=100) for line in combined_text.split("\n"))

    pdf = FPDF()
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if os.path.exists("DejaVuSans.ttf"):
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", "", 12)

    pdf.multi_cell(0, 10, wrapped_text)

    pdf.output(output_path)
    print(f"[✅] OCR text PDF saved to: {output_path}")

def combined_process(video_url, language="en"):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Transcribe audio
        language_code = "ar" if language == "ar" else None
        transcribed = transcribe_youtube_video(video_url, language_code, tmpdir)
        transcript = transcribed["transcript"]
        duration = transcribed["duration_minutes"]

        # Step 2: Download video for frame extraction
        video_path = os.path.join(tmpdir, "video.mp4")
        ydl_opts = {
            'format': 'best',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Step 3: Calculate dynamic frame interval:
        # shorter videos get more frames, longer fewer (min 3s, max 15s interval)
        frame_interval = max(1, min(10, duration / 20))  # more frequent snapshots

        # Step 4: Extract frames and filter blank images
        images = extract_frames_from_video(video_path, tmpdir, frame_interval_sec=frame_interval)

        # Step 5: OCR images asynchronously
        ocr_results = [None] * len(images)
        threads = []
        ocr_lang = "ara" if language == "ar" else "eng"

        for i, img_path in enumerate(images):
            t = threading.Thread(target=ocr_image_async, args=(img_path, ocr_results, i, ocr_lang))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        structured_outputs = []
        for img_path in images:
            structured_data = detect_structured_data(img_path)
            structured_outputs.append(structured_data)
            
        # ✅ Step 6.5: Save structured data to file
        os.makedirs("static", exist_ok=True)
        with open("static/structured_data.json", "w", encoding="utf-8") as f:
            json.dump(structured_outputs, f, ensure_ascii=False, indent=2)
            
        # Step 6: Combine transcript and OCR results
        ocr_texts = [txt for txt in ocr_results if txt and txt.strip()]
        save_ocr_text_to_pdf(ocr_texts, output_path="static/ocr_text.pdf")
        combined_text = transcript + "\n\n" + "\n\n".join(ocr_texts)

        # Step 7: Generate lessons and quiz
        course_data = generate_course_content(combined_text, language, video_duration_minutes=duration)

        # Step 8: Attach base64 images to output so frontend can show them if needed
        images_b64 = [image_to_base64(img) for img in images]

        course_data["extracted_images_base64"] = images_b64

        # Notes:
        # - OCR quality depends heavily on screenshot quality and font used in video.
        # - This process skips blank frames to reduce noise.
        # - You can adjust frame interval logic for your needs.
        # - Unlike PDF image extraction, video frames are images directly, so OCR is simpler here.

        return course_data