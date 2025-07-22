# tasks.py
from celery_app import celery

@celery.task(name='video.process_video')
def process_video_task(video_url):
    from app.processor import process_video
    return process_video(video_url)
