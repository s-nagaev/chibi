from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from chibi.config import application_settings

logger = logging.getLogger(__name__)


class ChibiScheduler:
    def __init__(self) -> None:
        if application_settings.redis:
            parsed = urlparse(application_settings.redis)
            job_store = RedisJobStore(
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip("/") or 0),
                password=parsed.password,
            )
            logger.info(f"Scheduler: using Redis job store ({parsed.hostname}:{parsed.port})")
        else:
            db_path = Path(application_settings.local_data_path) / "scheduler.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            job_store = SQLAlchemyJobStore(url=f"sqlite:///{db_path}")
            logger.info("Scheduler: using SQLite job store (%s)", db_path)

        self._scheduler = AsyncIOScheduler(jobstores={"default": job_store})

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self, wait: bool = True) -> None:
        self._scheduler.shutdown(wait=wait)

    def add_job(self, *args, **kwargs):
        return self._scheduler.add_job(*args, **kwargs)

    def remove_job(self, job_id: str) -> None:
        self._scheduler.remove_job(job_id)

    def get_jobs(self):
        return self._scheduler.get_jobs()
