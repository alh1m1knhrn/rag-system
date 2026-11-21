from typing import List, Set, Optional
from sqlalchemy.orm import Session
from sqlalchemy import delete

from src.db.models import ChatMessage, VisitedTopic, CompletedTask, UserState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ==================== История диалога ====================

def add_message(db: Session, user_id: str, role: str, content: str):
    msg = ChatMessage(user_id=user_id, role=role, content=content)
    db.add(msg)
    db.commit()


def get_chat_history(db: Session, user_id: str, limit: int = 20) -> List[BaseMessage]:
    """Возвращает последние сообщения в формате LangChain."""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()  # хронологический порядок
    messages = []
    for row in rows:
        if row.role == "human":
            messages.append(HumanMessage(content=row.content))
        else:
            messages.append(AIMessage(content=row.content))
    return messages


def clear_chat_history(db: Session, user_id: str):
    db.execute(delete(ChatMessage).where(ChatMessage.user_id == user_id))
    db.commit()


# ==================== Прогресс по главам ====================

def mark_topic_visited(db: Session, user_id: str, topic_key: str):
    exists = db.query(VisitedTopic).filter_by(user_id=user_id, topic_key=topic_key).first()
    if not exists:
        db.add(VisitedTopic(user_id=user_id, topic_key=topic_key))
        db.commit()


def get_visited_topics(db: Session, user_id: str) -> Set[str]:
    rows = db.query(VisitedTopic.topic_key).filter_by(user_id=user_id).all()
    return {r[0] for r in rows}


def clear_visited_topics(db: Session, user_id: str):
    db.execute(delete(VisitedTopic).where(VisitedTopic.user_id == user_id))
    db.commit()


# ==================== Выполненные задачи ====================

def mark_task_completed(db: Session, user_id: str, task_key: str):
    exists = db.query(CompletedTask).filter_by(user_id=user_id, task_key=task_key).first()
    if not exists:
        db.add(CompletedTask(user_id=user_id, task_key=task_key))
        db.commit()


def get_completed_tasks(db: Session, user_id: str) -> Set[str]:
    rows = db.query(CompletedTask.task_key).filter_by(user_id=user_id).all()
    return {r[0] for r in rows}


def clear_completed_tasks(db: Session, user_id: str):
    db.execute(delete(CompletedTask).where(CompletedTask.user_id == user_id))
    db.commit()


# ==================== Текущая задача ====================

def set_current_task(db: Session, user_id: str, task_key: Optional[str]):
    state = db.query(UserState).filter_by(user_id=user_id).first()
    if state:
        state.current_task_key = task_key
    else:
        state = UserState(user_id=user_id, current_task_key=task_key)
        db.add(state)
    db.commit()


def get_current_task(db: Session, user_id: str) -> Optional[str]:
    state = db.query(UserState).filter_by(user_id=user_id).first()
    return state.current_task_key if state else None


def clear_current_task(db: Session, user_id: str):
    set_current_task(db, user_id, None)


# ==================== Полная очистка пользователя ====================

def clear_all_user_data(db: Session, user_id: str):
    clear_chat_history(db, user_id)
    clear_visited_topics(db, user_id)
    clear_completed_tasks(db, user_id)
    clear_current_task(db, user_id)


# ==================== Результаты викторины ====================

def save_quiz_result(db: Session, user_id: str, nickname: str, score: int, time_seconds: int):
    from src.db.models import QuizResult
    result = QuizResult(
        user_id=user_id,
        nickname=nickname,
        score=score,
        time_seconds=time_seconds
    )
    db.add(result)
    db.commit()


def get_quiz_leaderboard(db: Session, limit: int = 10):
    from src.db.models import QuizResult
    from sqlalchemy import func

    # Лучший результат по каждому нику
    results = (
        db.query(QuizResult)
        .order_by(QuizResult.score.desc(), QuizResult.time_seconds.asc())
        .all()
    )

    seen = set()
    leaderboard = []
    for r in results:
        if r.nickname not in seen:
            seen.add(r.nickname)
            leaderboard.append((r.nickname, r.score, r.time_seconds))
            if len(leaderboard) >= limit:
                break
    return leaderboard


def upsert_user(db: Session, user_id: str, username: str = None, first_name: str = None, last_name: str = None):
    from src.db.models import BotUser
    user = db.query(BotUser).filter_by(user_id=user_id).first()
    if user:
        user.username = username
        user.first_name = first_name
        user.last_name = last_name
    else:
        user = BotUser(
            user_id=user_id,
            username=username,
            first_name=first_name,
            last_name=last_name
        )
        db.add(user)
    db.commit()


def get_all_users(db: Session):
    from src.db.models import BotUser
    return db.query(BotUser).order_by(BotUser.created_at).all()
