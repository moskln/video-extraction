from celery import Celery
from config import Config

# Set up Celery instance
celery = Celery('video_tasks', broker=Config.CELERY_BROKER_URL)
celery.conf.result_backend = Config.CELERY_RESULT_BACKEND

# Example async task
@celery.task(name='video.process_video')
def process_video_task(video_url):
    from app.processor import process_video  # Delay import to avoid loop
    return process_video(video_url)
