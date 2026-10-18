import pymupdf as fitz
import json
import os
base_path = r"raw/learnpython.pdf"
doc = fitz.open(base_path)
pages_data = []  #список для хранения джсон файла
for page_num, page in enumerate(doc,start=1):
    text = page.get_text()
    pages_data.append({
        "page_number": page_num,
        "text": text
    })
output_file = r"processed\parsed.json"
with open(output_file, "w",encoding="utf-8") as f:
    json.dump(pages_data, f , ensure_ascii=False, indent=4)
print(f"Успех! Сохранено {len(pages_data)} страниц в файл {output_file}")