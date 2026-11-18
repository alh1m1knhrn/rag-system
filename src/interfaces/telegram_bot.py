import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, FSInputFile
)
from aiogram.enums import ParseMode

from src.rag.chain import build_rag_chain
from src.config import BOT_TOKEN, GROQ_API_KEY, GROQ_MODEL
from src.code_executor.executor import execute_code, UnsafeCodeError
from tasks import TASKS, TOPICS_ORDER
from src.quiz.questions import QUESTIONS
import random
from langchain_openai import ChatOpenAI

from src.db.database import SessionLocal, init_db
from src.db import crud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

rag_chain = None

# Состояние квиза
# {user_id: {"questions": [...], "index": 0, "score": 0, "wrong_topics": set(), "nickname": str, "start_time": float}}
quiz_sessions = {}
# Ожидание ника: {user_id: True}
waiting_for_nickname = {}

TEXTBOOK_PATH = ROOT_DIR / "data" / "textbook.docx"

TOPICS = {
    "ch2": ("Глава 1. Переменные и простые типы данных", "Что такое переменные в Python? Как работать со строками, числами, списками и другими простыми типами данных?"),
    "ch3": ("Глава 2. Списки", "Что такое списки в Python? Как создавать списки, обращаться к элементам по индексу, изменять, добавлять и удалять элементы?"),
    "ch4": ("Глава 3. Работа со списками", "Как перебирать списки в цикле for? Как создавать числовые списки с помощью range? Что такое срезы списков?"),
    "ch5": ("Глава 4. Команды if", "Как работают условные операторы if, elif и else в Python? Как проверять условия и использовать их со списками?"),
    "ch6": ("Глава 5. Словари", "Что такое словари в Python? Как создавать словари, обращаться к значениям по ключам, перебирать словари и вкладывать их?"),
    "ch7": ("Глава 6. Ввод данных и циклы while", "Как работает функция input()? Как использовать циклы while? Как применять while со списками и словарями?"),
    "ch8": ("Глава 7. Функции", "Как определять и вызывать функции в Python? Как передавать аргументы, возвращать значения и работать со списками в функциях?"),
    "ch9": ("Глава 8. Классы", "Что такое классы в Python? Как создавать классы, работать с экземплярами, использовать наследование?"),

}

main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="❓ Задать вопрос"), KeyboardButton(text="💻 Выполнить код")],
        [KeyboardButton(text="📚 Разделы"), KeyboardButton(text="📝 Задачи")],
        [KeyboardButton(text="🍀 Испытать удачу"), KeyboardButton(text="🏆 Топ викторины")],
        [KeyboardButton(text="📥 Скачать учебник"), KeyboardButton(text="🗑️ Очистить контекст")],
        [KeyboardButton(text="🔄 Сбросить прогресс"), KeyboardButton(text="ℹ️ Помощь")],

    ],
    resize_keyboard=True
)


def get_topics_keyboard(user_id: str, highlight_key: str = None):
    db = SessionLocal()
    try:
        visited = crud.get_visited_topics(db, user_id)
    finally:
        db.close()
    buttons = []
    for key, (title, _) in TOPICS.items():
        short_title = title.replace("Глава ", "Гл. ")
        if key == highlight_key:
            short_title = "✨ " + short_title
        elif key in visited:
            short_title = "✅ " + short_title
        buttons.append([InlineKeyboardButton(text=short_title, callback_data=f"topic_{key}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_task_topics_keyboard():
    buttons = []
    for topic in TOPICS_ORDER:
        buttons.append([InlineKeyboardButton(text=topic, callback_data=f"tasktopic_{topic}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_tasks_keyboard(topic: str, user_id: str):
    db = SessionLocal()
    try:
        completed = crud.get_completed_tasks(db, user_id)
    finally:
        db.close()
    buttons = []
    for key, task in TASKS.items():
        if task["topic"] == topic:
            title = task["title"]
            if key in completed:
                title = "✅ " + title
            buttons.append([InlineKeyboardButton(text=title, callback_data=f"task_{key}")])
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="tasks_back")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я — RAG-ассистент по учебнику «Изучаем Python» Эрика Мэтиза.\n\n"
        "Могу отвечать на вопросы, выполнять код, показывать главы, давать задачи и отправлять учебник.",
        reply_markup=main_kb
    )


@dp.message(F.text == "ℹ️ Помощь")
@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📖 <b>Как пользоваться:</b>\n\n"
        "• Напиши вопрос — отвечу по учебнику.\n"
        "• «📚 Разделы» — выбери главу.\n"
        "• «📝 Задачи» — выбери тему и задачу, пришли решение кодом.\n"
        "• «💻 Выполнить код» — просто запустит код.\n"
        "• «📥 Скачать учебник» — получишь Word-файл.\n"
        "• «🍀 Испытать удачу» — викторина из 10 случайных вопросов.\n"
        "• «🗑️ Очистить контекст» — сбросит только историю диалога.\n"
        "• «🔄 Сбросить прогресс» — уберёт галочки глав и задач.",
        parse_mode=ParseMode.HTML,
        reply_markup=main_kb
    )


@dp.message(F.text == "🗑️ Очистить контекст")
@dp.message(Command("clear_context"))
async def cmd_clear_context(message: Message):
    user_id = str(message.from_user.id)
    db = SessionLocal()
    try:
        crud.clear_chat_history(db, user_id)
    finally:
        db.close()
    await message.answer("🗑️ История диалога очищена. Прогресс сохранён.", reply_markup=main_kb)


@dp.message(F.text == "🔄 Сбросить прогресс")
@dp.message(Command("reset_progress"))
async def cmd_reset_progress(message: Message):
    user_id = str(message.from_user.id)
    db = SessionLocal()
    try:
        crud.clear_visited_topics(db, user_id)
        crud.clear_completed_tasks(db, user_id)
        crud.clear_current_task(db, user_id)
    finally:
        db.close()
    await message.answer("🔄 Прогресс сброшен (галочки глав и задач убраны). История диалога сохранена.",
                         reply_markup=main_kb)


@dp.message(F.text == "📥 Скачать учебник")
async def send_textbook(message: Message):
    if not TEXTBOOK_PATH.exists():
        await message.answer("⚠️ Файл учебника не найден.\nПоложи `textbook.docx` в папку `data/`.",
                             reply_markup=main_kb)
        return
    await message.answer("Отправляю учебник...")
    document = FSInputFile(TEXTBOOK_PATH)
    await message.answer_document(document, caption="📘 Изучаем Python — Эрик Мэтиз", reply_markup=main_kb)


@dp.message(F.text == "📚 Разделы")
async def show_topics(message: Message):
    user_id = str(message.from_user.id)
    await message.answer("Выбери главу учебника:\n(✅ — уже просмотренные)", reply_markup=get_topics_keyboard(user_id))


@dp.message(F.text == "📝 Задачи")
async def show_task_topics(message: Message):
    await message.answer("Выбери тему задач:", reply_markup=get_task_topics_keyboard())


@dp.callback_query(F.data.startswith("tasktopic_"))
async def show_tasks_by_topic(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    topic = callback.data.replace("tasktopic_", "")
    await callback.message.edit_text(
        f"Задачи по теме «{topic}»:\n(✅ — выполненные)",
        reply_markup=get_tasks_keyboard(topic, user_id)
    )
    await callback.answer()


@dp.callback_query(F.data == "tasks_back")
async def tasks_back(callback: CallbackQuery):
    await callback.message.edit_text("Выбери тему задач:", reply_markup=get_task_topics_keyboard())
    await callback.answer()


@dp.callback_query(F.data.startswith("task_"))
async def show_task(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    task_key = callback.data.replace("task_", "")
    task = TASKS.get(task_key)
    if not task:
        await callback.answer("Задача не найдена")
        return

    db = SessionLocal()
    try:
        crud.set_current_task(db, user_id, task_key)
    finally:
        db.close()
    text = (
        f"📝 <b>{task['title']}</b>\n\n"
        f"{task['description']}\n\n"
        f"💡 Подсказка: <tg-spoiler>{task['hint']}</tg-spoiler>\n\n"
        f"Пришли код решения следующим сообщением."
    )
    await callback.message.answer(text, parse_mode=ParseMode.HTML)
    await callback.answer()


@dp.callback_query(F.data.startswith("topic_"))
async def process_topic(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    topic_key = callback.data.replace("topic_", "")
    topic_title, question = TOPICS.get(topic_key, ("Тема", "Объясни эту тему"))

    try:
        await callback.message.edit_reply_markup(reply_markup=get_topics_keyboard(user_id, highlight_key=topic_key))
        await asyncio.sleep(0.6)
    except Exception:
        pass

    db = SessionLocal()
    try:
        crud.mark_topic_visited(db, user_id, topic_key)
    finally:
        db.close()

    try:
        await callback.message.edit_reply_markup(reply_markup=get_topics_keyboard(user_id))
    except Exception:
        pass

    await callback.answer()
    await callback.message.answer(f"📖 <b>{topic_title}</b>\n\nИщу информацию в учебнике...", parse_mode=ParseMode.HTML)
    await bot.send_chat_action(chat_id=callback.message.chat.id, action="typing")

    # История теперь в БД, очищается только по кнопке

    try:
        config = {"configurable": {"session_id": user_id}}
        answer = await asyncio.to_thread(rag_chain.invoke, {"question": question}, config=config)
        await callback.message.answer(answer, parse_mode=None, reply_markup=main_kb)
    except Exception as e:
        logger.error(f"Topic error: {e}")
        await callback.message.answer("⚠️ Не удалось получить информацию по главе.попробуйте нажать кнопку очистить контекст", reply_markup=main_kb)


# Глобальный LLM для проверки задач
checker_llm = None


def get_checker_llm():
    global checker_llm
    if checker_llm is None:
        checker_llm = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
            max_tokens=600,
        )
    return checker_llm


async def check_solution(task_key: str, user_code: str) -> str:
    """Проверяет решение напрямую через LLM."""
    task = TASKS[task_key]
    llm = get_checker_llm()

    prompt = f"""Проверь решение студента.

Задача: {task['title']}
{task['description']}

Код студента:
{user_code}

Ответь строго в формате:
Вердикт: Правильно или Неправильно
Плюсы: кратко
Минусы: кратко
Эталон:
```python
код
```"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Check error: {e}")
        return "⚠️ Не удалось проверить решение. Попробуй ещё раз."


@dp.message(F.text == "💻 Выполнить код")
@dp.message(Command("code"))
async def cmd_code(message: Message):
    await message.answer("Отправь код, который нужно выполнить.")


# ==================== ВИКТОРИНА ====================

def get_quiz_keyboard(options):
    buttons = []
    for i, opt in enumerate(options):
        buttons.append([InlineKeyboardButton(text=opt, callback_data=f"quiz_{i}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(F.text == "🍀 Испытать удачу")
async def start_quiz(message: Message):
    user_id = str(message.from_user.id)
    waiting_for_nickname[user_id] = True
    await message.answer(
        "Введи ник для топа (только латинские буквы и цифры, 3–15 символов):",
        reply_markup=main_kb
    )


@dp.message(F.text == "🏆 Топ викторины")
async def show_leaderboard(message: Message):
    db = SessionLocal()
    try:
        rows = crud.get_quiz_leaderboard(db, limit=10)
    finally:
        db.close()

    if not rows:
        await message.answer("Топ пока пуст. Будь первым!", reply_markup=main_kb)
        return

    lines = ["🏆 <b>Топ викторины</b>\n"]
    for i, (nick, score, t) in enumerate(rows, 1):
        mins, secs = divmod(t, 60)
        time_str = f"{mins}:{secs:02d}" if mins else f"{secs} сек"
        lines.append(f"{i}. {nick} — {score}/10 ({time_str})")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=main_kb)


async def send_quiz_question(message_or_callback, user_id: str, edit: bool = False):
    session = quiz_sessions.get(user_id)
    if not session:
        return

    idx = session["index"]
    if idx >= 10:
        await finish_quiz(message_or_callback, user_id)
        return

    q = session["questions"][idx]
    text = f"Вопрос {idx + 1}/10\n\n{q['q']}"
    kb = get_quiz_keyboard(q["options"])

    if edit and hasattr(message_or_callback, "message"):
        await message_or_callback.message.edit_text(text, reply_markup=kb)
    else:
        await message_or_callback.answer(text, reply_markup=kb)


@dp.callback_query(F.data.startswith("quiz_"))
async def handle_quiz_answer(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    session = quiz_sessions.get(user_id)
    if not session:
        await callback.answer("Викторина не активна")
        return

    choice = int(callback.data.replace("quiz_", ""))
    q = session["questions"][session["index"]]
    correct = q["correct"]

    if choice == correct:
        session["score"] += 1
        await callback.answer("✅ Правильно!")
    else:
        session["wrong_topics"].add(q["topic"])
        correct_text = q["options"][correct]
        await callback.answer(f"❌ Неправильно. Верный ответ: {correct_text}", show_alert=True)

    session["index"] += 1
    await send_quiz_question(callback, user_id, edit=True)


async def finish_quiz(message_or_callback, user_id: str):
    session = quiz_sessions.pop(user_id, None)
    if not session:
        return

    # Считаем время
    import time
    elapsed = int(time.time() - session.get("start_time", time.time()))

    # Сохраняем результат
    db = SessionLocal()
    try:
        crud.save_quiz_result(
            db,
            user_id=user_id,
            nickname=session.get("nickname", "Аноним"),
            score=session["score"],
            time_seconds=elapsed
        )
        # Очищаем контекст
        crud.clear_chat_history(db, user_id)
    finally:
        db.close()

    score = session["score"]
    wrong = session["wrong_topics"]

    text = f"🏁 Викторина завершена!\n\nРезультат: {score}/10"

    if wrong:
        topics = ", ".join(sorted(wrong))
        text += f"\n\n📚 Стоит подтянуть темы:\n{topics}"
    else:
        text += "\n\n🎉 Отлично! Все темы на хорошем уровне."

    if hasattr(message_or_callback, "message"):
        await message_or_callback.message.edit_text(text)
        await message_or_callback.message.answer("Можешь начать новую викторину или продолжить обучение.",
                                                 reply_markup=main_kb)
    else:
        await message_or_callback.answer(text, reply_markup=main_kb)


@dp.message(F.text)
async def handle_message(message: Message):
    text = message.text.strip()
    user_id = str(message.from_user.id)

    # Ожидание ника для викторины
    if user_id in waiting_for_nickname:
        import re
        if re.fullmatch(r"[A-Za-z0-9]{3,15}", text):
            waiting_for_nickname.pop(user_id, None)
            n = min(10, len(QUESTIONS))
            if n == 0:
                await message.answer("В базе нет вопросов.", reply_markup=main_kb)
                return
            selected = random.sample(QUESTIONS, n)
            import time
            quiz_sessions[user_id] = {
                "questions": selected,
                "index": 0,
                "score": 0,
                "wrong_topics": set(),
                "nickname": text,
                "start_time": time.time()
            }
            await message.answer(f"Ник принят: {text}\nНачинаем викторину!")
            await send_quiz_question(message, user_id)
        else:
            await message.answer(
                "Ник должен содержать только латинские буквы и цифры (3–15 символов). Попробуй ещё раз:")
        return

    # Если у пользователя активна задача — проверяем решение
    db = SessionLocal()
    try:
        task_key = crud.get_current_task(db, user_id)
    finally:
        db.close()

    if task_key:
        await message.answer("🔍 Проверяю решение...")
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")

        # Выполняем код
        try:
            stdout, stderr, success = await asyncio.to_thread(execute_code, text)
            if success:
                if stdout.strip():
                    exec_info = f"Вывод:\n{stdout}"
                else:
                    # Функция определена, но не вызывалась
                    if "def " in text:
                        exec_info = "Вывод: функция определена успешно (вывода нет, так как функция не вызывалась)"
                    else:
                        exec_info = "Вывод: (нет вывода)"
            else:
                exec_info = f"Ошибка:\n{stderr or 'нет вывода'}"
        except Exception as e:
            logger.error(f"Execute error: {e}")
            exec_info = "Не удалось выполнить код."

        # Проверяем с помощью ИИ
        verdict = await check_solution(task_key, text)

        await message.answer(f"{verdict}\n\n{exec_info}", parse_mode=None, reply_markup=main_kb)

        # Если решение правильное — отмечаем как выполненное
        if "Правильно" in verdict and "Неправильно" not in verdict:
            db = SessionLocal()
            try:
                crud.mark_task_completed(db, user_id, task_key)
            finally:
                db.close()

        db = SessionLocal()
        try:
            crud.set_current_task(db, user_id, None)
        finally:
            db.close()
        return

    # Обычное выполнение кода
    if text.startswith("/code") or (
            len(text) > 15 and any(k in text for k in ("print(", "def ", "for ", "if ", "while "))):
        code = text.replace("/code", "", 1).strip().strip("`")
        if code.startswith("python"):
            code = code[6:].strip()

        if not code:
            await message.answer("Пришли код.", reply_markup=main_kb)
            return

        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        try:
            stdout, stderr, success = await asyncio.to_thread(execute_code, code)
            if success:
                result = f"✅ <b>Код выполнен</b>\n\n<pre>{stdout or '(нет вывода)'}</pre>"
            else:
                result = f"❌ <b>Ошибка</b>\n\n<pre>{stderr or stdout or 'Неизвестная ошибка'}</pre>"
            await message.answer(result, parse_mode=ParseMode.HTML, reply_markup=main_kb)
        except UnsafeCodeError as e:
            await message.answer(f"🚫 Запрещённый код: {e}", reply_markup=main_kb)
        except Exception as e:
            logger.error(f"Code error: {e}")
            await message.answer("⚠️ Не удалось выполнить код.", reply_markup=main_kb)
        return

    if text in ("❓ Задать вопрос",):
        await message.answer("Напиши свой вопрос.", reply_markup=main_kb)
        return

    # Обычный вопрос
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        config = {"configurable": {"session_id": user_id}}
        answer = await asyncio.to_thread(rag_chain.invoke, {"question": text}, config=config)
        await message.answer(answer, parse_mode=None, reply_markup=main_kb)
    except Exception as e:
        logger.error(f"RAG error: {e}")
        await message.answer("⚠️ Ошибка при обработке вопроса.", reply_markup=main_kb)


async def main():
    global rag_chain
    logger.info("Инициализация базы данных...")
    init_db()
    logger.info("Загружаем RAG...")
    rag_chain = build_rag_chain()
    logger.info("Бот запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
