"""
Кастомная история чата для LangChain, хранящая сообщения в SQLite.
"""
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.db.database import SessionLocal
from src.db import crud


class SQLChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        db = SessionLocal()
        try:
            return crud.get_chat_history(db, self.session_id)
        finally:
            db.close()

    def add_message(self, message: BaseMessage) -> None:
        role = "human" if isinstance(message, HumanMessage) else "ai"
        db = SessionLocal()
        try:
            crud.add_message(db, self.session_id, role, message.content)
        finally:
            db.close()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        db = SessionLocal()
        try:
            crud.clear_chat_history(db, self.session_id)
        finally:
            db.close()
