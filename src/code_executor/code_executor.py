from src.code_executor.executor import execute_code

stdout, stderr, success = execute_code("""
print("Hello")
print(2 + 2)
""")

print("OK:", success)
print("Вывод:", stdout)
print("Ошибки:", stderr)