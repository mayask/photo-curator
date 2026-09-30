from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_init_lock = Lock()


def init_database(database_url: str) -> Engine:
    global _engine, _session_factory
    with _init_lock:
        if _engine is not None:
            return _engine

        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False, "timeout": 30}

        _engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )

        if database_url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        _session_factory = sessionmaker(
            bind=_engine,
            expire_on_commit=False,
            autoflush=False,
        )
        Base.metadata.create_all(_engine)
        # create_all does not add a newly introduced index to an existing table.
        # Explicit check-first creation keeps lightweight upgrades safe without
        # requiring a separate migration service for index-only changes.
        for table in Base.metadata.sorted_tables:
            for index in table.indexes:
                index.create(_engine, checkfirst=True)
        return _engine


def reset_database_for_tests() -> None:
    global _engine, _session_factory
    with _init_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None


def session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        raise RuntimeError("Database has not been initialized")
    return _session_factory


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    with session_scope() as session:
        yield session


def database_is_healthy() -> bool:
    if _engine is None:
        return False
    try:
        with _engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
