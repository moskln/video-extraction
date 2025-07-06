# generator_content.py
import os
import json
import re
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_course_content(transcript, language="en", video_duration_minutes=10, log_func=print, translate_to=None):
    log_func(f"[INFO] Generating course content for {video_duration_minutes:.2f} minutes...")
    multiplier = max(1, math.ceil(video_duration_minutes / 10))
    num_lessons, num_questions = 4 * multiplier, 5 * multiplier

    prompt = f"""
You are an expert course designer.
Your task is to generate output in this language: {language}.

Given the transcript below, perform the following:
1. Summarize into {num_lessons} lessons with headers and explanations.
2. Extract motivational messages per lesson.
3. Generate {num_questions} quiz questions.

Return strictly JSON like:
```json
{{
  "lessons": [{{"title": "", "details": "", "motivational_message": ""}}],
  "quiz": [{{"question": "", "choices": [""], "answer": ""}}]
}}
"""
    log_func("[INFO] Sending prompt to OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + f'\n"""{transcript}"""'}],
        temperature=0.7
    )

    content = response.choices[0].message.content.strip()
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    json_str = match.group(1) if match else content

    try:
        result = json.loads(json_str)
        if translate_to and translate_to != language:
            log_func("[INFO] Translating lessons...")
            translated = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": f"Translate this JSON into {translate_to}:\n\n{json.dumps(result)}"}
                ],
                temperature=0.5
            )
            result = json.loads(translated.choices[0].message.content.strip())
        return result
    except Exception as e:
        log_func(f"[ERROR] Failed to parse JSON: {e}")
        return {"error": "Failed to parse GPT response", "raw": content}
