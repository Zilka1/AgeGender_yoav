"""Tests for .env loading (GENDER_LABEL_0/1, KAGGLE_* etc. are otherwise inert)."""

from __future__ import annotations

import os

from src.utils.config import load_env_file


def test_load_env_file_sets_new_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_VAR", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_VAR=hello\n")

    load_env_file(env_path)
    assert os.environ["SOME_TEST_VAR"] == "hello"
    del os.environ["SOME_TEST_VAR"]


def test_load_env_file_does_not_override_existing_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_VAR", "from_shell")
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_VAR=from_dotenv\n")

    load_env_file(env_path)
    assert os.environ["SOME_TEST_VAR"] == "from_shell"


def test_load_env_file_is_a_noop_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.env"
    load_env_file(missing_path)  # must not raise
