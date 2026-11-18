import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from src.config import BASE_DIR

# Путь к файлу БД (рядом с chroma_db или в data/)
DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "rag_bot.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # нужно для SQLite + aiogram
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Создаёт таблицы, если их ещё нет."""
    Base.metadata.create_all(bind=engine)
    print(f"База данных инициализирована: {DB_PATH}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
