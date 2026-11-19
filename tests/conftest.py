"""Общие фикстуры."""
import os
import sys
from pathlib import Path

# Устанавливаем фиктивные ключи ДО импорта модулей проекта
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("BOT_TOKEN", "test-token")

# Добавляем корень проекта в путь
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
