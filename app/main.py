# app/app.py

from flask import Flask, render_template, request, jsonify
from app.processor import combined_process
from app.logger import setup_logger
import traceback

logger = setup_logger()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    video_url = data.get("video_url")
    language = data.get("language", "en")

    logger.info(f"Incoming request: video_url={video_url}, language={language}")

    if not video_url:
        logger.error("No URL provided.")
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Handle YouTube Shorts format
        if "youtube.com/shorts/" in video_url:
            video_url = video_url.replace("youtube.com/shorts/", "youtube.com/watch?v=")
            logger.info(f"Normalized YouTube Shorts URL to: {video_url}")

        course_data = combined_process(video_url, language)
        logger.info(f"Processing succeeded. Frames: {len(course_data.get('extracted_images_base64', []))}")
        return jsonify(course_data)

    except Exception as e:
        logger.exception("Processing failed.")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True)
