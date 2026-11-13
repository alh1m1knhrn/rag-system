import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_store(input_json_path: str, db_directory: str):
    """
    Загружает чанки, создает эмбеддинги и сохраняет их в ChromaDB.
    """
    print(f"Загружаем чанки из: {input_json_path}")

    # 1. Загружаем наши нарезанные куски
    with open(input_json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # 2. Превращаем словари в объекты Document (стандартный формат LangChain)
    documents = []
    for chunk in chunks_data:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)

    print(f"Подготовлено {len(documents)} документов для индексации.")

    # 3. Инициализируем модель для создания эмбеддингов
    # Используем легкую и быструю мультиязычную модель, которая хорошо знает русский
    print("Загружаем модель эмбеддингов (при первом запуске она скачается из интернета)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Создаем векторную базу данных и сохраняем туда документы
    print(f"Создаем базу данных в папке: {db_directory}")
    print("Это может занять несколько минут, в зависимости от размера учебника...")

    # ВАЖНО: Если база уже существует, Chroma может выдать ошибку при повторном добавлении.
    # Поэтому мы удаляем старую базу, если она есть (только для этапа разработки).
    import shutil
    if os.path.exists(db_directory):
        print("Старая база найдена. Очищаем...")
        shutil.rmtree(db_directory)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_directory  # Здесь будут храниться файлы базы
    )

    print("Успех! Векторная база данных создана и сохранена.")
    return vectordb


# Блок для автономного запуска
if __name__ == "__main__":
    # Вычисляем корень проекта
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
    DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

    create_vector_store(INPUT_FILE, DB_DIR)