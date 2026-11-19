"""Тесты для базы данных (CRUD)."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db import crud
from src.db.models import ChatMessage, VisitedTopic, CompletedTask


@pytest.fixture
def db_session():
    """Создаёт временную in-memory БД для каждого теста."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_add_and_get_chat_history(db_session):
    crud.add_message(db_session, "user1", "human", "Привет")
    crud.add_message(db_session, "user1", "ai", "Здравствуй")

    history = crud.get_chat_history(db_session, "user1")
    assert len(history) == 2
    assert history[0].content == "Привет"
    assert history[1].content == "Здравствуй"


def test_clear_chat_history(db_session):
    crud.add_message(db_session, "user1", "human", "Тест")
    crud.clear_chat_history(db_session, "user1")

    history = crud.get_chat_history(db_session, "user1")
    assert len(history) == 0


def test_visited_topics(db_session):
    crud.mark_topic_visited(db_session, "user1", "ch3")
    crud.mark_topic_visited(db_session, "user1", "ch5")
    # Повторная отметка не должна дублироваться
    crud.mark_topic_visited(db_session, "user1", "ch3")

    topics = crud.get_visited_topics(db_session, "user1")
    assert topics == {"ch3", "ch5"}


def test_completed_tasks(db_session):
    crud.mark_task_completed(db_session, "user1", "lists_1")
    crud.mark_task_completed(db_session, "user1", "if_1")

    tasks = crud.get_completed_tasks(db_session, "user1")
    assert "lists_1" in tasks
    assert "if_1" in tasks
    assert len(tasks) == 2


def test_current_task(db_session):
    crud.set_current_task(db_session, "user1", "lists_1")
    assert crud.get_current_task(db_session, "user1") == "lists_1"

    crud.set_current_task(db_session, "user1", None)
    assert crud.get_current_task(db_session, "user1") is None


def test_clear_all_user_data(db_session):
    crud.add_message(db_session, "user1", "human", "Привет")
    crud.mark_topic_visited(db_session, "user1", "ch3")
    crud.mark_task_completed(db_session, "user1", "lists_1")
    crud.set_current_task(db_session, "user1", "lists_2")

    crud.clear_all_user_data(db_session, "user1")

    assert len(crud.get_chat_history(db_session, "user1")) == 0
    assert len(crud.get_visited_topics(db_session, "user1")) == 0
    assert len(crud.get_completed_tasks(db_session, "user1")) == 0
    assert crud.get_current_task(db_session, "user1") is None
