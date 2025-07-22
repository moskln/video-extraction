# test_task.py
from tasks import process_video_task

video_url = "https://www.youtube.com/shorts/PJlSDl_6eSk"
result = process_video_task.delay(video_url)

print("Task submitted. ID:", result.id)
print("Waiting for result...")

try:
    output = result.get(timeout=120)
    print("Task result:", output)
except Exception as e:
    print("Task failed:", (e))
