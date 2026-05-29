from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


def _database_path() -> Path:
    configured = Path(settings.sqlite_db_path)
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parent / configured


DATABASE_PATH = _database_path()
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


def enable_wal_mode() -> None:
    with engine.connect() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL"))
        connection.execute(text("PRAGMA synchronous=NORMAL"))
        connection.commit()


def init_db() -> None:
    import models  # noqa: F401 - registers SQLAlchemy models with Base metadata.

    Base.metadata.create_all(bind=engine)
    enable_wal_mode()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
