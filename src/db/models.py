from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint, ForeignKey
from sqlalchemy.orm import relationship

from src.db.database import Base


class ChatMessage(Base):
    """История диалога пользователя (для RAG)."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)          # "human" или "ai"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMessage user={self.user_id} role={self.role}>"


class VisitedTopic(Base):
    """Просмотренные главы (для галочек ✅)."""
    __tablename__ = "visited_topics"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    topic_key = Column(String, nullable=False)     # "ch3", "ch4", ...

    __table_args__ = (UniqueConstraint("user_id", "topic_key", name="uq_user_topic"),)


class CompletedTask(Base):
    """Выполненные задачи."""
    __tablename__ = "completed_tasks"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    task_key = Column(String, nullable=False)      # "lists_1", "if_2", ...

    __table_args__ = (UniqueConstraint("user_id", "task_key", name="uq_user_task"),)


class UserState(Base):
    """Текущее состояние пользователя (текущая задача)."""
    __tablename__ = "user_states"

    user_id = Column(String, primary_key=True)
    current_task_key = Column(String, nullable=True)


class QuizResult(Base):
    """Результаты викторины для топа."""
    __tablename__ = "quiz_results"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    nickname = Column(String, nullable=False)
    score = Column(Integer, nullable=False)          # 0-10
    time_seconds = Column(Integer, nullable=False)   # время прохождения
    created_at = Column(DateTime, default=datetime.utcnow)


class BotUser(Base):
    """Пользователи бота (сохраняются навсегда)."""
    __tablename__ = "bot_users"

    user_id = Column(String, primary_key=True)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
