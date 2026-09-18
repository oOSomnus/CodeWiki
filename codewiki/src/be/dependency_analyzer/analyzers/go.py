"""Tree-sitter based analyzer for Go source files.

The analyzer deliberately exposes the same small seam as the other language
adapters in this package: one function returns ``Node`` and
``CallRelationship`` records for a single file.  Name resolution remains
best-effort; repository-wide matching is performed by ``CallGraphAnalyzer``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

from tree_sitter import Node as TreeNode
from tree_sitter_language_pack import get_parser

from codewiki.src.be.dependency_analyzer.models.core import CallRelationship, Node

logger = logging.getLogger(__name__)


GO_BUILTIN_TYPES = frozenset(
    {
        "any",
        "bool",
        "byte",
        "complex64",
        "complex128",
        "error",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "rune",
        "string",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
    }
)


def _text(node: TreeNode | None) -> str:
    if node is None:
        return ""
    value = node.text
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _children(node: TreeNode | None) -> Iterable[TreeNode]:
    if node is None:
        return ()
    return node.named_children


def _walk(node: TreeNode | None) -> Iterable[TreeNode]:
    if node is None:
        return
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _field(node: TreeNode, name: str) -> TreeNode | None:
    return node.child_by_field_name(name)


def _line(node: TreeNode) -> int:
    return node.start_point[0] + 1


def _source(lines: list[str], node: TreeNode) -> str:
    start = node.start_point[0]
    end = min(node.end_point[0] + 1, len(lines))
    return "\n".join(lines[start:end])


def _clean_comment(value: str) -> str:
    value = value.strip()
    if value.startswith("//"):
        return value[2:].lstrip()
    if value.startswith("/*") and value.endswith("*/"):
        value = value[2:-2]
    lines = []
    for line in value.splitlines():
        line = line.strip()
        if line.startswith("*"):
            line = line[1:].lstrip()
        lines.append(line)
    return "\n".join(lines).strip()


def _doc_comment(node: TreeNode) -> str:
    def collect(previous: TreeNode | None) -> list[str]:
        comments: list[str] = []
        while previous is not None and previous.type == "comment":
            comments.append(_clean_comment(_text(previous)))
            previous = previous.prev_sibling
        comments.reverse()
        return comments

    comments = collect(node.prev_sibling)
    if not comments and node.parent is not None:
        comments = collect(node.parent.prev_sibling)
    return "\n".join(part for part in comments if part).strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {'"', "`"} and value[-1] == value[0]:
        return value[1:-1]
    return value


class TreeSitterGoAnalyzer:
    """Extract Go declarations and syntax-level relationships."""

    def __init__(self, file_path: str, content: str, repo_path: str | None = None):
        self.file_path = Path(file_path)
        self.content = content
        self.repo_path = repo_path or ""
        self.relative_path = self._get_relative_path()
        self.lines = content.splitlines()
        self.package_name = ""
        self.imports: dict[str, str] = {}
        self.import_paths: dict[str, str] = {}
        self.nodes: list[Node] = []
        self.symbols: dict[str, Node] = {}
        self.type_names: set[str] = set()
        self.call_relationships: list[CallRelationship] = []
        self._relationship_keys: set[tuple[str, str]] = set()
        self._pending_type_edges: list[tuple[str, TreeNode, int, set[str]]] = []
        self._bodies: list[tuple[TreeNode, str, dict[str, str], str | None]] = []
        self._parse()

    def _get_relative_path(self) -> str:
        if self.repo_path:
            try:
                path = os.path.relpath(str(self.file_path), self.repo_path)
            except ValueError:
                path = str(self.file_path)
        else:
            path = str(self.file_path)
        return path.replace("\\", "/")

    def _component_id(self, logical_name: str) -> str:
        return f"{self.relative_path}::{logical_name}"

    def _qualified_name(self, logical_name: str) -> str:
        return f"{self.package_name}.{logical_name}" if self.package_name else logical_name

    def _add_node(
        self,
        node: TreeNode,
        logical_name: str,
        component_type: str,
        *,
        node_type: str | None = None,
        class_name: str | None = None,
        parameters: list[str] | None = None,
        base_classes: list[str] | None = None,
    ) -> Node:
        component_id = self._component_id(logical_name)
        existing = self.symbols.get(logical_name)
        if existing is not None:
            return existing

        docstring = _doc_comment(node)
        result = Node(
            id=component_id,
            name=logical_name,
            component_type=component_type,
            file_path=str(self.file_path),
            relative_path=self.relative_path,
            source_code=_source(self.lines, node),
            start_line=_line(node),
            end_line=node.end_point[0] + 1,
            has_docstring=bool(docstring),
            docstring=docstring,
            parameters=parameters,
            node_type=node_type or component_type,
            base_classes=base_classes or None,
            class_name=class_name,
            display_name=f"{component_type} {logical_name}",
            component_id=component_id,
            language="go",
            qualified_name=self._qualified_name(logical_name),
        )
        self.nodes.append(result)
        self.symbols[logical_name] = result
        if component_type in {"struct", "interface", "type"}:
            self.type_names.add(logical_name)
        return result

    def _parse(self) -> None:
        try:
            tree = get_parser("go").parse(self.content.encode("utf-8"))
            root = tree.root_node
            self._read_package_and_imports(root)
            self._collect_declarations(root)
            self._resolve_pending_type_edges()
            self._collect_relationships()
        except Exception:  # noqa: BLE001 — one malformed file must not abort a sweep
            logger.exception("Error parsing Go file %s", self.file_path)

    def _read_package_and_imports(self, root: TreeNode) -> None:
        for child in root.named_children:
            if child.type == "package_clause":
                package = next((item for item in _children(child) if item.type == "package_identifier"), None)
                self.package_name = _text(package)
            elif child.type == "import_declaration":
                for spec in _walk(child):
                    if spec.type != "import_spec":
                        continue
                    path = _unquote(_text(_field(spec, "path")))
                    if not path:
                        continue
                    alias = _text(_field(spec, "name"))
                    if not alias:
                        alias = path.rsplit("/", 1)[-1]
                    if alias not in {"_", "."}:
                        self.imports[alias] = alias
                    self.import_paths[alias] = path

    def _collect_declarations(self, root: TreeNode) -> None:
        for child in root.named_children:
            if child.type == "type_declaration":
                self._collect_type_declaration(child)
            elif child.type in {"const_declaration", "var_declaration"}:
                self._collect_value_declaration(child)
            elif child.type == "function_declaration":
                self._collect_function(child)
            elif child.type == "method_declaration":
                self._collect_method(child)

    def _collect_type_declaration(self, declaration: TreeNode) -> None:
        for child in declaration.named_children:
            if child.type not in {"type_spec", "type_alias"}:
                continue
            name = _text(_field(child, "name"))
            type_node = _field(child, "type")
            if not name or type_node is None:
                continue
            if child.type == "type_alias":
                component_type = "type"
                node_type = "type_alias"
            elif type_node.type == "struct_type":
                component_type = "struct"
                node_type = "struct"
            elif type_node.type == "interface_type":
                component_type = "interface"
                node_type = "interface"
            else:
                component_type = "type"
                node_type = "type"

            base_classes = self._embedded_types(type_node)
            component = self._add_node(
                child,
                name,
                component_type,
                node_type=node_type,
                base_classes=base_classes,
            )
            self._pending_type_edges.append((component.id, type_node, _line(child), {name}))
            if type_node.type == "interface_type":
                self._collect_interface_methods(name, type_node)

    def _embedded_types(self, type_node: TreeNode) -> list[str] | None:
        if type_node.type not in {"struct_type", "interface_type"}:
            return None
        result: list[str] = []
        for child in _walk(type_node):
            if child.type != "field_declaration" and child.type != "type_elem":
                continue
            field_name = _field(child, "name")
            if field_name is not None:
                continue
            target = _field(child, "type") or next(iter(child.named_children), None)
            for reference in self._type_references(target):
                if reference not in result:
                    result.append(reference)
        return result or None

    def _collect_interface_methods(self, owner: str, interface_node: TreeNode) -> None:
        for child in interface_node.named_children:
            if child.type != "method_elem":
                continue
            name = _text(_field(child, "name"))
            if not name:
                continue
            parameters_node = _field(child, "parameters")
            parameters = self._parameters(parameters_node)
            method = self._add_node(
                child,
                f"{owner}.{name}",
                "method",
                node_type="method",
                class_name=owner,
                parameters=parameters,
            )
            if parameters_node is not None:
                self._pending_type_edges.append((method.id, parameters_node, _line(child), set()))
            result = _field(child, "result")
            if result is not None:
                self._pending_type_edges.append((method.id, result, _line(child), set()))

    def _collect_value_declaration(self, declaration: TreeNode) -> None:
        kind = "const" if declaration.type == "const_declaration" else "variable"
        for spec in declaration.named_children:
            if spec.type not in {"const_spec", "var_spec"}:
                continue
            names_node = _field(spec, "name")
            names = [
                _text(item)
                for item in _walk(names_node)
                if item.type in {"identifier", "field_identifier"}
            ]
            if not names:
                names = [_text(names_node)] if _text(names_node) else []
            type_node = _field(spec, "type")
            value_node = _field(spec, "value")
            for name in dict.fromkeys(names):
                component = self._add_node(
                    spec,
                    name,
                    "variable",
                    node_type=kind,
                )
                if type_node is not None:
                    self._pending_type_edges.append((component.id, type_node, _line(spec), {name}))
                if value_node is not None:
                    self._bodies.append((value_node, component.id, {}, None))

    def _collect_function(self, declaration: TreeNode) -> None:
        name = _text(_field(declaration, "name"))
        if not name:
            return
        parameters_node = _field(declaration, "parameters")
        result = self._add_node(
            declaration,
            name,
            "function",
            node_type="function",
            parameters=self._parameters(parameters_node),
        )
        if parameters_node is not None:
            self._pending_type_edges.append((result.id, parameters_node, _line(declaration), set()))
        result_node = _field(declaration, "result")
        if result_node is not None:
            self._pending_type_edges.append((result.id, result_node, _line(declaration), set()))
        body = _field(declaration, "body")
        if body is not None:
            self._bodies.append((body, result.id, self._parameter_types(parameters_node), None))

    def _collect_method(self, declaration: TreeNode) -> None:
        receiver = _field(declaration, "receiver")
        receiver_name, receiver_variable, receiver_type = self._receiver_info(receiver)
        name = _text(_field(declaration, "name"))
        if not name or not receiver_name:
            return
        logical_name = f"{receiver_name}.{name}"
        parameters_node = _field(declaration, "parameters")
        component = self._add_node(
            declaration,
            logical_name,
            "method",
            node_type="method",
            class_name=receiver_name,
            parameters=self._parameters(parameters_node),
        )
        if parameters_node is not None:
            self._pending_type_edges.append((component.id, parameters_node, _line(declaration), set()))
        result_node = _field(declaration, "result")
        if result_node is not None:
            self._pending_type_edges.append((component.id, result_node, _line(declaration), set()))
        context = self._parameter_types(parameters_node)
        if receiver_variable and receiver_type:
            context[receiver_variable] = receiver_type
        body = _field(declaration, "body")
        if body is not None:
            self._bodies.append((body, component.id, context, receiver_name))

    def _receiver_info(self, receiver: TreeNode | None) -> tuple[str, str, str]:
        parameter = next((item for item in _walk(receiver) if item.type == "parameter_declaration"), None)
        if parameter is None:
            return "", "", ""
        variable = _text(_field(parameter, "name"))
        type_node = _field(parameter, "type")
        type_text = _text(type_node)
        receiver_name = self._base_type(type_text)
        return receiver_name, variable, receiver_name

    def _parameters(self, parameters: TreeNode | None) -> list[str] | None:
        if parameters is None:
            return None
        result = [
            _text(child).strip()
            for child in parameters.named_children
            if child.type == "parameter_declaration" and _text(child).strip()
        ]
        return result or None

    def _parameter_types(self, parameters: TreeNode | None) -> dict[str, str]:
        result: dict[str, str] = {}
        if parameters is None:
            return result
        for parameter in parameters.named_children:
            if parameter.type != "parameter_declaration":
                continue
            type_name = self._base_type(_text(_field(parameter, "type")))
            names_node = _field(parameter, "name")
            names = [
                _text(item)
                for item in _walk(names_node)
                if item.type in {"identifier", "field_identifier"}
            ]
            if not names and _text(names_node):
                names = [_text(names_node)]
            for name in names:
                if name and type_name:
                    result[name] = type_name
        return result

    def _resolve_pending_type_edges(self) -> None:
        for caller, type_node, line, excluded in self._pending_type_edges:
            for reference in self._type_references(type_node):
                if reference in excluded:
                    continue
                self._add_reference(caller, reference, line)

    def _type_references(self, node: TreeNode | None) -> list[str]:
        if node is None:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in _walk(node):
            if item.type == "qualified_type":
                value = _text(item)
            elif item.type == "type_identifier":
                value = _text(item)
            else:
                continue
            value = value.strip()
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _collect_relationships(self) -> None:
        for body, caller, context, receiver_type in self._bodies:
            for item in _walk(body):
                if item.type == "call_expression":
                    function = _field(item, "function")
                    raw = _text(function)
                    if raw:
                        self._add_call(caller, raw, _line(item), context, receiver_type)
                elif item.type == "composite_literal":
                    self._add_type_node_reference(caller, _field(item, "type"), _line(item))
                elif item.type == "type_assertion_expression":
                    self._add_type_node_reference(caller, _field(item, "type"), _line(item))

    def _add_call(
        self,
        caller: str,
        raw: str,
        line: int,
        context: dict[str, str],
        receiver_type: str | None,
    ) -> None:
        raw = re.sub(r"\[[^\]]*\]$", "", raw.strip())
        raw = raw.replace(" ", "")
        if not raw:
            return
        if "." in raw:
            root, rest = raw.split(".", 1)
            if root in context:
                candidate = f"{context[root]}.{rest}"
                resolved = self._resolve_reference(candidate)
                if resolved is not None:
                    self._append_relationship(caller, resolved, line, True)
                    return
            if receiver_type and root in {"self", "this"}:
                candidate = f"{receiver_type}.{rest}"
                resolved = self._resolve_reference(candidate)
                if resolved is not None:
                    self._append_relationship(caller, resolved, line, True)
                    return
        reference = self._normalize_reference(raw)
        resolved = self._resolve_reference(reference)
        if resolved is not None:
            self._append_relationship(caller, resolved, line, True)
        else:
            self._append_relationship(caller, reference, line, False)

    def _add_type_node_reference(self, caller: str, node: TreeNode | None, line: int) -> None:
        for reference in self._type_references(node):
            self._add_reference(caller, reference, line)

    def _add_reference(self, caller: str, raw: str, line: int) -> None:
        reference = self._normalize_reference(raw)
        if self._base_type(reference) in GO_BUILTIN_TYPES:
            return
        resolved = self._resolve_reference(reference)
        if resolved is not None:
            self._append_relationship(caller, resolved, line, True)
        else:
            self._append_relationship(caller, reference, line, False)

    def _normalize_reference(self, raw: str) -> str:
        value = raw.strip().replace("::", ".")
        value = re.sub(r"\s+", "", value)
        value = value.rstrip("!")
        root, separator, rest = value.partition(".")
        if root in self.import_paths:
            imported = self.import_paths[root].replace("/", ".")
            return f"{imported}.{rest}" if separator else imported
        return value

    def _base_type(self, value: str) -> str:
        value = value.strip()
        value = re.sub(r"^[*\[\]]+", "", value)
        value = value.split("[", 1)[0]
        return value.rsplit(".", 1)[-1]

    def _resolve_reference(self, reference: str) -> str | None:
        value = self._normalize_reference(reference)
        if self.package_name and value.startswith(self.package_name + "."):
            value = value[len(self.package_name) + 1 :]
        node = self.symbols.get(value)
        return node.id if node is not None else None

    def _append_relationship(self, caller: str, callee: str, line: int, resolved: bool) -> None:
        key = (caller, callee)
        if key in self._relationship_keys:
            return
        self._relationship_keys.add(key)
        self.call_relationships.append(
            CallRelationship(caller=caller, callee=callee, call_line=line, is_resolved=resolved)
        )


def analyze_go_file(
    file_path: str, content: str, repo_path: str | None = None
) -> tuple[list[Node], list[CallRelationship]]:
    """Analyze one Go file and return documentable nodes and relationships."""

    analyzer = TreeSitterGoAnalyzer(file_path, content, repo_path)
    return analyzer.nodes, analyzer.call_relationships
