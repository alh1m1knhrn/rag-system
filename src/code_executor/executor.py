import ast
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Tuple


class CodeExecutionError(Exception):
    pass


class UnsafeCodeError(Exception):
    pass


# Список запрещённых модулей и функций
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests",
    "urllib", "pathlib", "builtins", "eval", "exec", "compile",
    "open", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr"
}

FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__code__", "__closure__"
}


def check_code_safety(code: str) -> None:
    """Проверяет код на наличие опасных конструкций через AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeExecutionError(f"Синтаксическая ошибка: {e}")

    for node in ast.walk(tree):
        # Запрещённые имена
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Запрещённое имя: {node.id}")

        # Запрещённые атрибуты
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise UnsafeCodeError(f"Запрещённый атрибут: {node.attr}")

        # Импорты
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []

            for name in names:
                if name in FORBIDDEN_NAMES:
                    raise UnsafeCodeError(f"Запрещённый импорт: {name}")


def execute_code(code: str, timeout: int = 5) -> Tuple[str, str, bool]:
    """
    Выполняет код безопасно.

    Returns:
        (stdout, stderr, success)
    """
    check_code_safety(code)

    # Создаём временный файл
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),  # ограничиваем рабочую директорию
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        success = result.returncode == 0

        return stdout, stderr, success

    except subprocess.TimeoutExpired:
        return "", f"Превышено время выполнения ({timeout} сек)", False
    except Exception as e:
        return "", f"Ошибка выполнения: {e}", False
    finally:
        # Удаляем временный файл
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# Пример использования
if __name__ == "__main__":
    test_code = """
print("Привет из песочницы!")
for i in range(3):
    print(i)
"""
    out, err, ok = execute_code(test_code)
    print("SUCCESS:", ok)
    print("STDOUT:", out)
    print("STDERR:", err)
