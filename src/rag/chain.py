import os
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_chroma import Chroma
from transformers import pipeline, AutoTokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.rag.prompts import PROMPT_TEMPLATE, CONDENSE_QUESTION_PROMPT


def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Стр. {page}] {doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(db_directory: str):
    print("1. Загружаем модель эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("2. Подключаемся к базе данных...")
    vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    print("3. Загружаем языковую модель (Qwen3-4B) в 4-битном квантовании...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    model_id = "Qwen/Qwen3-4B-Instruct-2507"

    # 1. Создаем конфигурацию для 4-битного квантования (экономит память)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 2. Загружаем саму модель с этой конфигурацией
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. Создаем пайплайн, передавая в него уже загруженную модель
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        # enable_thinking=False  # <-- Эту строку лучше убрать, для Instruct-версии она не нужна
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    chat_model = ChatHuggingFace(llm=llm)

    print("4. Собираем цепочку с историей...")

    # Цепочка для переформулировки вопроса
    condense_chain = CONDENSE_QUESTION_PROMPT | chat_model | StrOutputParser()

    def contextualize_question(inputs):
        if not inputs.get("chat_history"):
            return inputs["question"]

        new_question = condense_chain.invoke({
            "question": inputs["question"],
            "chat_history": inputs["chat_history"]
        }).strip()

        # Убираем лишние кавычки
        new_question = new_question.replace('"', '').replace("'", "").strip()

        # 🚨 ДЕТЕКТОР ГАЛЛЮЦИНАЦИЙ:
        # Если модель написала код, длинный текст, или начала отвечать ("Конечно", "Лямбда" и т.д.)
        if len(new_question) > 80 or "```" in new_question or "Конечно" in new_question:
            print(f"⚠️ [Отладка] Модель начала отвечать вместо переформулировки. Включаем ручной режим.")

            # Достаем предыдущий вопрос пользователя из истории
            previous_question = ""
            for msg in reversed(inputs.get("chat_history", [])):
                if msg.type == "human":
                    previous_question = msg.content
                    break

            # Склеиваем предыдущий вопрос и текущий для поиска
            combined = f"{previous_question} {inputs['question']}"
            print(f"🔄 [Отладка] Ищем в базе по ручному запросу: '{combined}'")
            return combined

        # Если модель справилась — используем её вариант
        print(f"🔄 [Отладка] Ищем в базе по запросу: '{new_question}'")
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
            | chat_model
            | StrOutputParser()
    )

    # Хранилище истории в памяти (для реального проекта лучше использовать БД)
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Оборачиваем в менеджер истории
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversational_rag_chain


# Блок для проверки (остается без изменений)
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
    chain = build_rag_chain(DB_DIR)

    session_id = "student_1"
    config = {"configurable": {"session_id": session_id}}
    questions = [
        "Что такое цикл while?",
        "Приведи пример кода с ним",
        "А какие аргументы он принимает?"
    ]
    for q in questions:
        print(f"\n👤 Пользователь: {q}")
        answer = chain.invoke({"question": q}, config=config)
        print(f"🤖 Агент: {answer}")