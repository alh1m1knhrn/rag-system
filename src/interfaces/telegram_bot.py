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
from langchain_openai import ChatOpenAI

from src.db.database import SessionLocal, init_db
from src.db import crud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

rag_chain = None

TEXTBOOK_PATH = ROOT_DIR / "data" / "textbook.docx"


TOPICS = {
    "ch3": ("Глава 3. Списки", "Что такое списки в Python? Как создавать списки, обращаться к элементам по индексу, изменять, добавлять и удалять элементы?"),
    "ch4": ("Глава 4. Работа со списками", "Как перебирать списки в цикле for? Как создавать числовые списки с помощью range? Что такое срезы списков?"),
    "ch5": ("Глава 5. Команды if", "Как работают условные операторы if, elif и else в Python? Как проверять условия и использовать их со списками?"),
    "ch6": ("Глава 6. Словари", "Что такое словари в Python? Как создавать словари, обращаться к значениям по ключам, перебирать словари и вкладывать их?"),
    "ch7": ("Глава 7. Ввод данных и циклы while", "Как работает функция input()? Как использовать циклы while? Как применять while со списками и словарями?"),
    "ch8": ("Глава 8. Функции", "Как определять и вызывать функции в Python? Как передавать аргументы, возвращать значения и работать со списками в функциях?"),
    "ch9": ("Глава 9. Классы", "Что такое классы в Python? Как создавать классы, работать с экземплярами, использовать наследование?"),
    "ch10": ("Глава 10. Файлы и исключения", "Как читать из файлов и записывать в файлы в Python? Как обрабатывать исключения с помощью try-except?"),
    "ch11": ("Глава 11. Тестирование", "Как тестировать функции и классы в Python? Как писать тесты?"),
}

main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="❓ Задать вопрос"), KeyboardButton(text="💻 Выполнить код")],
        [KeyboardButton(text="📚 Разделы"), KeyboardButton(text="📝 Задачи")],
        [KeyboardButton(text="📥 Скачать учебник"), KeyboardButton(text="🧹 Очистить историю")],
        [KeyboardButton(text="ℹ️ Помощь")],
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
        "• «🧹 Очистить историю» — сбросит контекст и отметки.",
        parse_mode=ParseMode.HTML,
        reply_markup=main_kb
    )


@dp.message(F.text == "🧹 Очистить историю")
@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    user_id = str(message.from_user.id)
    db = SessionLocal()
    try:
        crud.clear_all_user_data(db, user_id)
    finally:
        db.close()
    await message.answer("🧹 История диалога, отметки глав и выполненных задач очищены.", reply_markup=main_kb)


@dp.message(F.text == "📥 Скачать учебник")
async def send_textbook(message: Message):
    if not TEXTBOOK_PATH.exists():
        await message.answer("⚠️ Файл учебника не найден.\nПоложи `textbook.docx` в папку `data/`.", reply_markup=main_kb)
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
        await callback.message.answer("⚠️ Не удалось получить информацию по главе.", reply_markup=main_kb)


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


@dp.message(F.text)
async def handle_message(message: Message):
    text = message.text.strip()
    user_id = str(message.from_user.id)

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
            exec_info = f"Вывод:\n{stdout}" if success else f"Ошибка:\n{stderr or 'нет вывода'}"
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
    if text.startswith("/code") or (len(text) > 15 and any(k in text for k in ("print(", "def ", "for ", "if ", "while "))):
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
