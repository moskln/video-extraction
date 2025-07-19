from flask import Flask, render_template, request, jsonify
from processor import combined_process
from app.logger import setup_logger

logger = setup_logger()

logger.info("Video processing started.")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Your frontend HTML here

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    video_url = data.get("video_url")
    language = data.get("language", "en")

    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        course_data = combined_process(video_url, language)
        return jsonify(course_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)