from __future__ import annotations

from datetime import datetime, timedelta

from PIL import Image, ImageDraw
from sqlalchemy import func, select

from app.database import init_database, session_scope
from app.models import Collection, Job, Photo
from app.worker import WorkerService


def test_full_job_scans_analyzes_and_curates_local_library(settings):
    start = datetime(2023, 7, 14, 10, 0)
    for index in range(6):
        image = Image.new("RGB", (640, 420), (50 + index * 20, 100, 150))
        ImageDraw.Draw(image).rectangle((80 + index * 3, 70, 500, 350), outline="white", width=8)
        exif = Image.Exif()
        exif[36867] = (start + timedelta(minutes=index * 5)).strftime("%Y:%m:%d %H:%M:%S")
        image.save(settings.photo_root / f"photo-{index}.jpg", exif=exif, quality=90)

    init_database(settings.database_url)
    worker = WorkerService(settings)
    job_id = worker.enqueue("full")
    assert worker._claim_next_job() == job_id
    worker._execute_job(job_id)

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.status == "completed"
        assert session.scalar(select(func.count()).select_from(Photo)) == 6
        assert (
            session.scalar(
                select(func.count()).select_from(Photo).where(Photo.analysis_status == "done")
            )
            == 6
        )
        assert session.scalar(select(func.count()).select_from(Collection)) >= 2
        assert all(photo.thumb_path for photo in session.scalars(select(Photo)))

    assert not list(settings.temp_dir.iterdir())
