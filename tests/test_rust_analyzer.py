"""Tests for the tree-sitter based Rust analyzer."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_language_pack")

from codewiki.src.be.dependency_analyzer.analyzers.rust import analyze_rust_file
from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
from codewiki.src.be.dependency_analyzer.ast_parser import DependencyParser


SAMPLE = """\
use crate::service::{helper as h, Client};
use std::fmt::Debug;

mod service {
    pub fn helper() {}
}

/// A worker that forwards events to a service.
struct Worker {
    client: Client,
}

trait Service {
    fn run(&self, input: Input);
}

impl Service for Worker {
    fn run(&self, input: Input) {
        self.validate();
        h();
        println!("processed");
    }
}

impl Worker {
    fn validate(&self) {}
}
"""


def _analyze(tmp_path: Path):
    file_path = tmp_path / "lib.rs"
    file_path.write_text(SAMPLE, encoding="utf-8")
    return analyze_rust_file(str(file_path), SAMPLE, repo_path=str(tmp_path))


def test_extracts_rust_declarations_and_metadata(tmp_path: Path) -> None:
    nodes, _ = _analyze(tmp_path)
    by_name = {node.name: node for node in nodes}

    assert by_name["service"].component_type == "module"
    assert by_name["service.helper"].component_type == "function"
    assert by_name["Worker"].component_type == "struct"
    assert by_name["Worker"].has_docstring
    assert "worker that forwards" in by_name["Worker"].docstring.lower()
    assert by_name["Service"].component_type == "interface"
    assert by_name["Service.run"].component_type == "method"
    assert by_name["Worker.run"].component_type == "method"
    assert by_name["Worker.validate"].component_type == "method"
    assert by_name["Service.run"].parameters == ["&self", "input: Input"]
    assert all(node.language == "rust" for node in nodes)
    assert by_name["Worker"].id == "lib.rs::Worker"
    assert by_name["Worker.run"].id == "lib.rs::Worker.run"


def test_extracts_rust_type_impl_and_call_relationships(tmp_path: Path) -> None:
    _, relationships = _analyze(tmp_path)
    edges = {(rel.caller, rel.callee, rel.is_resolved) for rel in relationships}

    assert ("lib.rs::Worker", "crate.service.Client", False) in edges
    assert ("lib.rs::Worker", "lib.rs::Service", True) in edges
    assert ("lib.rs::Worker.run", "lib.rs::Worker.validate", True) in edges
    assert ("lib.rs::Worker.run", "lib.rs::service.helper", True) in edges
    assert ("lib.rs::Worker.run", "println", False) in edges


def test_resolves_impl_declared_before_type(tmp_path: Path) -> None:
    source = """\
impl Service for Worker { fn run(&self) {} }
trait Service { fn run(&self); }
struct Worker;
"""
    file_path = tmp_path / "lib.rs"
    file_path.write_text(source, encoding="utf-8")

    _, relationships = analyze_rust_file(str(file_path), source, repo_path=str(tmp_path))
    edges = {(rel.caller, rel.callee, rel.is_resolved) for rel in relationships}

    assert ("lib.rs::Worker", "lib.rs::Service", True) in edges


def test_call_graph_filters_rust_runtime_noise(tmp_path: Path) -> None:
    file_path = tmp_path / "lib.rs"
    file_path.write_text(SAMPLE, encoding="utf-8")
    result = CallGraphAnalyzer().analyze_code_files(
        [{"path": "lib.rs", "name": "lib.rs", "extension": ".rs", "language": "rust"}],
        str(tmp_path),
    )

    edges = {(rel["caller"], rel["callee"]) for rel in result["relationships"]}
    assert ("lib.rs::Worker.run", "println") not in edges
    assert ("lib.rs::Worker.run", "lib.rs::Worker.validate") in edges
    assert "rust" in result["call_graph"]["languages_found"]


def test_dependency_parser_resolves_rust_module_cross_file_calls(tmp_path: Path) -> None:
    (tmp_path / "service.rs").write_text("pub fn helper() {}\n", encoding="utf-8")
    (tmp_path / "lib.rs").write_text(
        "mod service;\nfn run() { service::helper(); }\n", encoding="utf-8"
    )

    components = DependencyParser(str(tmp_path), use_gitignore=False).parse_repository()

    assert "service.rs::helper" in components
    assert "lib.rs::run" in components
    assert "service.rs::helper" in components["lib.rs::run"].depends_on
