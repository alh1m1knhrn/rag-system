"""Тесты базы вопросов викторины."""
from src.quiz.questions import QUESTIONS


def test_questions_count():
    assert len(QUESTIONS) >= 50, "Должно быть достаточно вопросов"


def test_question_structure():
    for q in QUESTIONS:
        assert "q" in q
        assert "options" in q
        assert "correct" in q
        assert "topic" in q
        assert len(q["options"]) == 4
        assert 0 <= q["correct"] <= 3


def test_topics_exist():
    topics = {q["topic"] for q in QUESTIONS}
    assert len(topics) >= 5


def test_no_empty_questions():
    for q in QUESTIONS:
        assert q["q"].strip()
        assert all(opt.strip() for opt in q["options"])
