import os
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.rag.prompts import PROMPT_TEMPLATE, CONDENSE_QUESTION_PROMPT
from src.config import GROQ_API_KEY, GROQ_MODEL, CHROMA_DB_PATH
from src.db.chat_history import SQLChatMessageHistory


def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Стр. {page}] {doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(db_directory: str = None):
    if db_directory is None:
        db_directory = CHROMA_DB_PATH

    print("1. Загружаем модель эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("2. Подключаемся к базе данных...")
    vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    print("3. Подключаемся к Groq API...")
    llm = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=2048,
    )

    print("4. Собираем цепочку с историей...")

    # Цепочка для переформулировки вопроса
    condense_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    def contextualize_question(inputs):
        chat_history = inputs.get("chat_history") or []
        question = inputs["question"]

        # Ограничиваем историю последними 4 сообщениями
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]

        if not chat_history:
            return question

        try:
            new_question = condense_chain.invoke({
                "question": question,
                "chat_history": chat_history
            }).strip()

            # Чистим ответ
            new_question = new_question.replace('"', '').replace("'", "").replace("`", "").strip()
            # Убираем возможные префиксы
            for prefix in ("Вопрос:", "Переформулированный вопрос:", "Ответ:"):
                if new_question.startswith(prefix):
                    new_question = new_question[len(prefix):].strip()

            # Если переформулировка слишком длинная или подозрительная — берём оригинал
            if (len(new_question) > 150 or
                len(new_question) < 5 or
                "```" in new_question or
                new_question.lower().startswith(("конечно", "ответ", "вот"))):
                print(f"⚠️ [Отладка] Плохая переформулировка, используем оригинал: '{question}'")
                return question

            print(f"🔄 [Отладка] Ищем по: '{new_question}'")
            return new_question

        except Exception as e:
            print(f"⚠️ Ошибка переформулировки: {e}, используем оригинал")
            return question

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=contextualize_question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"])),
            question=lambda x: x["standalone_question"]
        )
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    def get_session_history(session_id: str):
        return SQLChatMessageHistory(session_id)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversational_rag_chain


# Тестовый запуск
if __name__ == "__main__":
    from src.db.database import init_db
    init_db()

    chain = build_rag_chain()

    session_id = "student_1"
    config = {"configurable": {"session_id": session_id}}

    questions = [
        "Что такое цикл while?",
        "Приведи пример кода с ним",
        "как предотвратить его зацикливание"
    ]

    for q in questions:
        print(f"\n👤 Пользователь: {q}")
        answer = chain.invoke({"question": q}, config=config)
        print(f"🤖 Агент: {answer}")
