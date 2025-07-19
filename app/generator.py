from openai import OpenAI
import os
import json
import re
import math
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_course_content(transcript, language="en", video_duration_minutes=10):
    print(f"[INFO] Generating course content for {video_duration_minutes} minutes of video...")

    multiplier = max(1, math.ceil(video_duration_minutes / 10))

    base_lessons = 4
    base_quiz_questions = 5

    num_lessons = base_lessons * multiplier
    num_questions = base_quiz_questions * multiplier

    prompt = f"""
You are an expert course designer.

Your task is to generate output in this language: {language}.

Given the transcript below, perform the following:

1. Provide a detailed and structured summary of the content into {num_lessons} key lessons with clear headers and detailed explanations.
2. For each lesson, extract any motivational, instructional, or spiritual messages.
3. Then, generate {num_questions} quiz questions (multiple choice or true/false) based on the content.

Format your response as a JSON code block like:
```json
{{
  "lessons": [
    {{ "title": "Lesson Title", "details": "Full explanation here.", "motivational_message": "Motivational or spiritual insight." }}
  ],
  "quiz": [
    {{
      "question": "...?",
      "choices": ["A", "B", "C", "D"],
      "answer": "Correct answer"
    }}
  ]
}}

Only return valid JSON. No explanation outside the JSON.
"""

    print(f"[INFO] Sending request to OpenAI with {num_lessons} lessons and {num_questions} questions...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + f'\n"""{transcript}"""'}],
        temperature=0.7
    )

    content = response.choices[0].message.content.strip()

    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    json_str = match.group(1) if match else content

    try:
        data = json.loads(json_str)
        print(f"[INFO] Successfully parsed GPT response with {len(data.get('lessons', []))} lessons and {len(data.get('quiz', []))} questions.")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to parse GPT response: {e}")
        return {"error": "Failed to parse GPT response as JSON.", "raw": content}