import sys
import os
sys.path.append(os.path.dirname(__file__))
from celery import Celery
from dotenv import load_dotenv


load_dotenv()

celery = Celery(
    'video_tasks',
    broker=os.environ.get('CELERY_BROKER_URL'),
    backend=os.environ.get('CELERY_RESULT_BACKEND')
)

import tasks  # 👈 Required to register @celery.task decorators
