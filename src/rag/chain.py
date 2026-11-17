import os
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.rag.prompts import PROMPT_TEMPLATE, CONDENSE_QUESTION_PROMPT
from src.config import GROQ_API_KEY, GROQ_MODEL, CHROMA_DB_PATH

# Общее хранилище истории
history_store = {}


def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]


def clear_session_history(session_id: str):
    """Очищает историю диалога пользователя."""
    if session_id in history_store:
        history_store[session_id].clear()


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
        max_tokens=1024,
    )

    print("4. Собираем цепочку с историей...")

    # Цепочка для переформулировки вопроса
    condense_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    def contextualize_question(inputs):
        if not inputs.get("chat_history"):
            return inputs["question"]

        new_question = condense_chain.invoke({
            "question": inputs["question"],
            "chat_history": inputs["chat_history"]
        }).strip()

        # Убираем лишние кавычки
        new_question = new_question.replace('"', '').replace("'", "").strip()

        # Детектор галлюцинаций / плохой переформулировки
        if len(new_question) > 120 or "```" in new_question or "Конечно" in new_question or "Ответ:" in new_question:
            print(f"⚠️ [Отладка] Плохая переформулировка. Используем ручной режим.")
            previous_question = ""
            for msg in reversed(inputs.get("chat_history", [])):
                if msg.type == "human":
                    previous_question = msg.content
                    break
            combined = f"{previous_question} {inputs['question']}".strip()
            print(f"🔄 [Отладка] Ищем по: '{combined}'")
            return combined

        print(f"🔄 [Отладка] Ищем по: '{new_question}'")
        return new_question

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

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversational_rag_chain


# Тестовый запуск
if __name__ == "__main__":
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
