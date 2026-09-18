"""Tests for language registration outside the AST adapters."""

from pathlib import Path

from codewiki.cli.utils.repo_validator import SUPPORTED_EXTENSIONS, count_code_files
from codewiki.cli.utils.validation import detect_supported_languages


def test_cli_recognizes_go_and_rust_files(tmp_path: Path) -> None:
    (tmp_path / "main.go").write_text("package main\n", encoding="utf-8")
    (tmp_path / "lib.rs").write_text("fn main() {}\n", encoding="utf-8")

    assert ".go" in SUPPORTED_EXTENSIONS
    assert ".rs" in SUPPORTED_EXTENSIONS
    assert count_code_files(tmp_path) == 2

    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "generated.rs").write_text("fn ignored() {}\n", encoding="utf-8")

    assert detect_supported_languages(tmp_path) == [("Go", 1), ("Rust", 1)]
