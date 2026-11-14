import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = r"C:\Users\Xiaomi Book Pro 2026\PycharmProjects\rag_python_tutor\data\chroma_db"

print(f"Ищем базу по пути: {DB_DIR}")

# Проверяем, существует ли папка с базой
if not os.path.exists(DB_DIR):
    print("❌ ОШИБКА: Папка с базой данных не найдена! Вы точно запускали embedder.py?")
    exit()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# ВАЖНАЯ ПРОВЕРКА: Смотрим, сколько кусков текста лежит в базе
db_count = vectordb._collection.count()
print(f"✅ База загружена! Количество фрагментов (чанков) в ней: {db_count}\n")

if db_count == 0:
    print("❌ База пуста! Вам нужно вернуться и заново запустить embedder.py")
    exit()

test_questions = [
     "Как объявить функцию?",
    "Что такое итератор?",
    "Как создать список в Python?",
    "В чем разница между списком и кортежем?",
    "Что такое lambda-функция?",
    "Как работает цикл while?"]

for q in test_questions:
    print(f"\n🔍 ВОПРОС: {q}")
    results = vectordb.similarity_search_with_score(q, k=3)

    # Проверяем, нашлось ли хоть что-то
    if not results:
        print("  ⚠️ Ничего не найдено. Возможно, база пуста.")
        continue

    for i, (doc, score) in enumerate(results):
        page = doc.metadata.get("page", "?")
        text_snippet = doc.page_content.replace("\n", " ")[:200]

        print(f"  [{i + 1}] Стр. {page} (Дистанция: {score:.4f})")
        print(f"      Текст: {text_snippet}...")