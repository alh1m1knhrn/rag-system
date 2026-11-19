"""Тесты безопасного выполнения кода."""
import pytest
from src.code_executor.executor import execute_code, UnsafeCodeError, CodeExecutionError


def test_simple_print():
    stdout, stderr, success = execute_code('print("Hello")')
    assert success is True
    assert "Hello" in stdout


def test_syntax_error():
    with pytest.raises(CodeExecutionError):
        execute_code("print(")


def test_forbidden_import():
    with pytest.raises(UnsafeCodeError):
        execute_code("import os")


def test_forbidden_name():
    with pytest.raises(UnsafeCodeError):
        execute_code("eval('1+1')")


def test_timeout():
    stdout, stderr, success = execute_code("while True: pass", timeout=1)
    assert success is False
    assert "время" in stderr.lower() or "timeout" in stderr.lower()


def test_function_no_output():
    code = """
def add(a, b):
    return a + b
"""
    stdout, stderr, success = execute_code(code)
    assert success is True
    assert stdout == ""
