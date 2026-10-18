import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(input_json_path: str, output_json_path: str):
    """
    Читает распарсенный JSON, разбивает текст на чанки
    и сохраняет с метаданными в новый JSON.
    """
    print(f"Загружаем данные из: {input_json_path}")

    # 1. Читаем наш распарсенный файл
    with open(input_json_path, "r", encoding="utf-8") as f:
        pages_data = json.load(f)

    # 2. Настраиваем сплиттер (нарезчик текста)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Размер куска (символов)
        chunk_overlap=150,  # Перекрытие, чтобы не терять смысл на стыках
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []

    # 3. Нарезаем текст постранично
    for page in pages_data:
        page_text = page.get("text", "")
        page_num = page.get("page_number", "unknown")
        source = page.get("source", "learnpython.pdf")

        # Пропускаем пустые страницы
        if not page_text.strip():
            continue

        # Разбиваем текст одной страницы на куски
        page_chunks = text_splitter.split_text(page_text)

        # 4. Сохраняем каждый кусок вместе с метаданными
        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "page": page_num,
                    "source": source
                }
            })

    # 5. Создаем папку, если её нет, и сохраняем результат
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"Успех! Получено {len(all_chunks)} чанков.")
    print(f"Сохранено в: {output_json_path}")


# Блок для автономного запуска
if __name__ == "__main__":
    # Вычисляем корень проекта, сначала находить абсолютный путь,git push -u origin main а затем отрезает файл до корня
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Обратите внимание: я указал ваше имя файла parsed.json!
    INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "parsed.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")

    create_chunks(INPUT_FILE, OUTPUT_FILE)