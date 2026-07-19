"""
backend/db/session.py
----------------------
SQLAlchemy session factory.
DATABASE_URL in .env controls the backend:
  - sqlite:///./mechsage.db  (dev default)
  - postgresql+psycopg2://...  (prod)
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.core.config import settings
from backend.db.models import Base

# SQLite needs check_same_thread=False for FastAPI's async usage
_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    echo=False,         # set True to log all SQL in debug mode
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def create_tables() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
