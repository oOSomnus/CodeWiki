"""Tests for the tree-sitter based Go analyzer."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_language_pack")

from codewiki.src.be.dependency_analyzer.analyzers.go import analyze_go_file
from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
from codewiki.src.be.dependency_analyzer.ast_parser import DependencyParser


SAMPLE = """\
package pipeline

import (
    format "fmt"
    svc "example.com/project/service"
)

// Buffers events before forwarding them downstream.
type Buffer struct {
    svc.Client
    capacity int
}

type Flushable interface {
    Flush() error
}

func (b *Buffer) Push(item Item) error {
    validate(item)
    format.Println(item)
    svc.Process(item)
    return nil
}

func validate(item Item) error {
    return nil
}
"""


def _analyze(tmp_path: Path):
    file_path = tmp_path / "buffer.go"
    file_path.write_text(SAMPLE, encoding="utf-8")
    return analyze_go_file(str(file_path), SAMPLE, repo_path=str(tmp_path))


def test_extracts_go_declarations_and_metadata(tmp_path: Path) -> None:
    nodes, _ = _analyze(tmp_path)
    by_name = {node.name: node for node in nodes}

    assert by_name["Buffer"].component_type == "struct"
    assert by_name["Buffer"].has_docstring
    assert "Buffers events" in by_name["Buffer"].docstring
    assert by_name["Flushable"].component_type == "interface"
    assert by_name["Flushable.Flush"].component_type == "method"
    assert by_name["Buffer.Push"].component_type == "method"
    assert by_name["Buffer.Push"].parameters == ["item Item"]
    assert by_name["validate"].component_type == "function"
    assert all(node.language == "go" for node in nodes)
    assert by_name["Buffer"].id == "buffer.go::Buffer"
    assert by_name["Buffer.Push"].id == "buffer.go::Buffer.Push"


def test_extracts_go_type_and_call_relationships(tmp_path: Path) -> None:
    _, relationships = _analyze(tmp_path)
    edges = {(rel.caller, rel.callee, rel.is_resolved) for rel in relationships}

    assert ("buffer.go::Buffer", "example.com.project.service.Client", False) in edges
    assert ("buffer.go::Buffer.Push", "Item", False) in edges
    assert ("buffer.go::Buffer.Push", "buffer.go::validate", True) in edges
    assert ("buffer.go::Buffer.Push", "fmt.Println", False) in edges
    assert ("buffer.go::Buffer.Push", "example.com.project.service.Process", False) in edges


def test_call_graph_filters_go_standard_library_noise(tmp_path: Path) -> None:
    file_path = tmp_path / "buffer.go"
    file_path.write_text(SAMPLE, encoding="utf-8")
    result = CallGraphAnalyzer().analyze_code_files(
        [{"path": "buffer.go", "name": "buffer.go", "extension": ".go", "language": "go"}],
        str(tmp_path),
    )

    edges = {(rel["caller"], rel["callee"]) for rel in result["relationships"]}
    assert ("buffer.go::Buffer.Push", "fmt.Println") not in edges
    assert ("buffer.go::Buffer.Push", "buffer.go::validate") in edges
    assert "go" in result["call_graph"]["languages_found"]


def test_dependency_parser_resolves_go_cross_file_calls(tmp_path: Path) -> None:
    (tmp_path / "base.go").write_text("package pipeline\nfunc Helper() {}\n", encoding="utf-8")
    (tmp_path / "main.go").write_text(
        "package pipeline\nfunc Run() { Helper() }\n", encoding="utf-8"
    )

    components = DependencyParser(str(tmp_path), use_gitignore=False).parse_repository()

    assert "base.go::Helper" in components
    assert "main.go::Run" in components
    assert "base.go::Helper" in components["main.go::Run"].depends_on
