"""Tree-sitter based analyzer for Rust source files."""

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


RUST_PRIMITIVE_TYPES = frozenset(
    {
        "bool",
        "char",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "str",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "()",
        "!",
    }
)


def _text(node: TreeNode | None) -> str:
    if node is None:
        return ""
    value = node.text
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
        value = value[2:]
    elif value.startswith("/*") and value.endswith("*/"):
        value = value[2:-2]
    lines = []
    for line in value.splitlines():
        line = line.strip()
        if line.startswith("*"):
            line = line[1:].lstrip()
        lines.append(line)
    return "\n".join(lines).strip()


def _doc_comment(node: TreeNode) -> str:
    comments: list[str] = []
    previous = node.prev_sibling
    while previous is not None and previous.type in {"line_comment", "block_comment"}:
        value = _clean_comment(_text(previous))
        if value.startswith("/"):
            value = value[1:].lstrip()
        comments.append(value)
        previous = previous.prev_sibling
    comments.reverse()
    return "\n".join(part for part in comments if part).strip()


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "({[<":
            depth += 1
        elif char in ")}]>":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


class TreeSitterRustAnalyzer:
    """Extract Rust declarations and syntax-level relationships."""

    def __init__(self, file_path: str, content: str, repo_path: str | None = None):
        self.file_path = Path(file_path)
        self.content = content
        self.repo_path = repo_path or ""
        self.relative_path = self._get_relative_path()
        self.file_module = self._get_file_module()
        self.lines = content.splitlines()
        self.imports: dict[str, str] = {}
        self.nodes: list[Node] = []
        self.symbols: dict[str, Node] = {}
        self.qualified_symbols: dict[str, Node] = {}
        self.call_relationships: list[CallRelationship] = []
        self._relationship_keys: set[tuple[str, str]] = set()
        self._pending_type_edges: list[
            tuple[str, TreeNode, int, set[str], tuple[str, ...], str | None]
        ] = []
        self._pending_impl_edges: list[tuple[str, str, int, tuple[str, ...]]] = []
        self._bodies: list[
            tuple[TreeNode, str, tuple[str, ...], str | None, dict[str, str]]
        ] = []
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

    def _get_file_module(self) -> tuple[str, ...]:
        path = self.relative_path.rsplit(".", 1)[0]
        parts = [part for part in path.split("/") if part and part != "."]
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] in {"lib", "main", "mod"}:
            parts = parts[:-1]
        return tuple(parts)

    def _component_id(self, logical_name: str) -> str:
        return f"{self.relative_path}::{logical_name}"

    def _qualified_name(self, logical_name: str) -> str:
        parts = ("crate", *self.file_module, *[part for part in logical_name.split(".") if part])
        return ".".join(parts)

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
            language="rust",
            qualified_name=self._qualified_name(logical_name),
        )
        self.nodes.append(result)
        self.symbols[logical_name] = result
        self.qualified_symbols[result.qualified_name or ""] = result
        return result

    def _parse(self) -> None:
        try:
            tree = get_parser("rust").parse(self.content.encode("utf-8"))
            root = tree.root_node
            self._read_uses(root)
            self._collect_declarations(root.named_children, ())
            self._resolve_pending_type_edges()
            self._resolve_pending_impl_edges()
            self._collect_relationships()
        except Exception:  # noqa: BLE001 — one malformed file must not abort a sweep
            logger.exception("Error parsing Rust file %s", self.file_path)

    def _read_uses(self, root: TreeNode) -> None:
        for node in _walk(root):
            if node.type == "use_declaration":
                self._record_use_text(_text(node))

    def _record_use_text(self, value: str) -> None:
        value = value.strip().rstrip(";").strip()
        value = re.sub(r"^(?:pub(?:\([^)]*\))?\s+)?use\s+", "", value)
        self._record_use_clause(value)

    def _record_use_clause(self, clause: str, prefix: str = "") -> None:
        clause = clause.strip()
        if not clause:
            return
        if clause.startswith("{") and clause.endswith("}"):
            for item in _split_top_level(clause[1:-1]):
                self._record_use_clause(item, prefix)
            return

        group_start = clause.find("::{")
        if group_start >= 0 and clause.endswith("}"):
            base = clause[:group_start]
            group = clause[group_start + 2 :]
            combined = f"{prefix}::{base}" if prefix else base
            self._record_use_clause(group, combined.strip(":"))
            return

        alias_match = re.match(r"^(.+?)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$", clause)
        if alias_match:
            path, alias = alias_match.groups()
        else:
            path, alias = clause, clause.rsplit("::", 1)[-1]
        if alias == "*":
            return
        if prefix:
            path = f"{prefix}::{path}" if path else prefix
        self.imports[alias] = self._normalize_path(path)

    def _collect_declarations(self, children: Iterable[TreeNode], scope: tuple[str, ...]) -> None:
        for node in children:
            if node.type == "mod_item":
                self._collect_module(node, scope)
            elif node.type == "struct_item":
                self._collect_struct(node, scope)
            elif node.type == "enum_item":
                self._collect_enum(node, scope)
            elif node.type == "trait_item":
                self._collect_trait(node, scope)
            elif node.type == "impl_item":
                self._collect_impl(node, scope)
            elif node.type in {"function_item", "function_signature_item"}:
                self._collect_function(node, scope, None)
            elif node.type == "type_item":
                self._collect_type_alias(node, scope)
            elif node.type in {"const_item", "static_item"}:
                self._collect_value(node, scope)

    def _collect_module(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        name = _text(_field(node, "name"))
        if not name:
            return
        logical = self._logical(scope, name)
        self._add_node(node, logical, "module", node_type="module")
        body = _field(node, "body")
        if body is not None:
            self._collect_declarations(body.named_children, (*scope, name))

    def _collect_struct(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        name = _text(_field(node, "name"))
        if not name:
            return
        component = self._add_node(node, self._logical(scope, name), "struct", node_type="struct")
        generic_names = self._generic_names(node)
        body = _field(node, "body")
        if body is not None:
            self._pending_type_edges.append(
                (component.id, body, _line(node), generic_names, scope, name)
            )

    def _collect_enum(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        name = _text(_field(node, "name"))
        if not name:
            return
        component = self._add_node(node, self._logical(scope, name), "enum", node_type="enum")
        body = _field(node, "body")
        if body is not None:
            self._pending_type_edges.append((component.id, body, _line(node), set(), scope, name))

    def _collect_trait(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        name = _text(_field(node, "name"))
        if not name:
            return
        bounds = _field(node, "bounds")
        base_classes = self._type_references(bounds)
        component = self._add_node(
            node,
            self._logical(scope, name),
            "interface",
            node_type="trait",
            base_classes=base_classes or None,
        )
        if bounds is not None:
            self._pending_type_edges.append((component.id, bounds, _line(node), {name}, scope, name))
        body = _field(node, "body")
        if body is not None:
            for child in body.named_children:
                if child.type in {"function_signature_item", "function_item"}:
                    self._collect_function(child, scope, name)

    def _collect_impl(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        type_node = _field(node, "type")
        owner_type = self._base_type(_text(type_node))
        if not owner_type:
            return
        trait_node = _field(node, "trait")
        trait = _text(trait_node)
        if trait:
            # Resolve after all declarations have been collected so an impl
            # that appears before its struct still gets an implementation edge.
            self._pending_impl_edges.append((owner_type, trait, _line(node), scope))
        body = _field(node, "body")
        if body is not None:
            for child in body.named_children:
                if child.type in {"function_item", "function_signature_item"}:
                    self._collect_function(child, scope, owner_type)

    def _collect_function(
        self,
        node: TreeNode,
        scope: tuple[str, ...],
        owner_type: str | None,
    ) -> None:
        name = _text(_field(node, "name"))
        if not name:
            return
        logical_parts = (*scope, owner_type, name) if owner_type else (*scope, name)
        logical = ".".join(part for part in logical_parts if part)
        parameters_node = _field(node, "parameters")
        component = self._add_node(
            node,
            logical,
            "method" if owner_type else "function",
            node_type="method" if owner_type else "function",
            class_name=owner_type,
            parameters=self._parameters(parameters_node),
        )
        if parameters_node is not None:
            self._pending_type_edges.append(
                (component.id, parameters_node, _line(node), set(), scope, owner_type)
            )
        return_type = _field(node, "return_type")
        if return_type is not None:
            self._pending_type_edges.append(
                (component.id, return_type, _line(node), set(), scope, owner_type)
            )
        body = _field(node, "body")
        if body is not None:
            context = self._parameter_types(parameters_node)
            self._bodies.append((body, component.id, scope, owner_type, context))

    def _collect_type_alias(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        name = _text(_field(node, "name"))
        type_node = _field(node, "type")
        if not name:
            return
        component = self._add_node(node, self._logical(scope, name), "type", node_type="type_alias")
        if type_node is not None:
            self._pending_type_edges.append((component.id, type_node, _line(node), {name}, scope, name))

    def _collect_value(self, node: TreeNode, scope: tuple[str, ...]) -> None:
        name = _text(_field(node, "name"))
        if not name:
            return
        kind = "const" if node.type == "const_item" else "static"
        component = self._add_node(node, self._logical(scope, name), "variable", node_type=kind)
        type_node = _field(node, "type")
        if type_node is not None:
            self._pending_type_edges.append((component.id, type_node, _line(node), {name}, scope, None))
        value = _field(node, "value")
        if value is not None:
            self._bodies.append((value, component.id, scope, None, {}))

    def _logical(self, scope: tuple[str, ...], name: str) -> str:
        return ".".join((*scope, name))

    def _generic_names(self, node: TreeNode) -> set[str]:
        result = set()
        for item in _walk(_field(node, "type_parameters")):
            if item.type == "type_parameter":
                name = _text(_field(item, "name"))
                if name:
                    result.add(name)
        return result

    def _parameters(self, parameters: TreeNode | None) -> list[str] | None:
        if parameters is None:
            return None
        result = [
            _text(child).strip()
            for child in parameters.named_children
            if child.type in {"parameter", "self_parameter"} and _text(child).strip()
        ]
        return result or None

    def _parameter_types(self, parameters: TreeNode | None) -> dict[str, str]:
        result: dict[str, str] = {}
        if parameters is None:
            return result
        for child in parameters.named_children:
            if child.type == "parameter":
                pattern = _field(child, "pattern")
                type_node = _field(child, "type")
                name = _text(pattern)
                type_name = self._base_type(_text(type_node))
                if name and type_name:
                    result[name] = type_name
        return result

    def _resolve_pending_type_edges(self) -> None:
        for caller, node, line, excluded, scope, owner_type in self._pending_type_edges:
            for reference in self._type_references(node):
                if reference in excluded:
                    continue
                self._add_reference(caller, reference, line, scope, owner_type)

    def _resolve_pending_impl_edges(self) -> None:
        for owner_type, trait, line, scope in self._pending_impl_edges:
            caller = self._resolve_reference(owner_type, scope, owner_type)
            if caller is None:
                continue
            reference = self._normalize_reference(trait, scope, None)
            resolved = self._resolve_reference(reference, scope, None)
            self._append_relationship(caller, resolved or reference, line, resolved is not None)

    def _type_references(self, node: TreeNode | None) -> list[str]:
        result: list[str] = []

        def visit(current: TreeNode) -> None:
            if current.type == "primitive_type":
                return
            if current.type in {"type_identifier", "scoped_type_identifier"}:
                value = _text(current).strip()
                if value and value not in result:
                    result.append(value)
                return
            for child in current.named_children:
                visit(child)

        if node is not None:
            visit(node)
        return result

    def _collect_relationships(self) -> None:
        for body, caller, scope, owner_type, context in self._bodies:
            self._walk_body(body, caller, scope, owner_type, context)

    def _walk_body(
        self,
        node: TreeNode,
        caller: str,
        scope: tuple[str, ...],
        owner_type: str | None,
        context: dict[str, str],
    ) -> None:
        if node.type == "let_declaration":
            type_node = _field(node, "type")
            if type_node is not None:
                for reference in self._type_references(type_node):
                    self._add_reference(caller, reference, _line(node), scope, owner_type)
                pattern = _field(node, "pattern")
                name = _text(pattern)
                type_name = self._base_type(_text(type_node))
                if name and type_name:
                    context[name] = type_name

        if node.type == "call_expression":
            function = _field(node, "function")
            raw = _text(function)
            if raw:
                self._add_call(caller, raw, _line(node), scope, owner_type, context)
            type_arguments = _field(node, "type_arguments")
            if type_arguments is not None:
                for reference in self._type_references(type_arguments):
                    self._add_reference(caller, reference, _line(node), scope, owner_type)
        elif node.type == "macro_invocation":
            macro = _field(node, "macro")
            raw = _text(macro)
            if not raw:
                raw = next((_text(child) for child in node.named_children if child.type == "identifier"), "")
            if raw:
                self._add_call(caller, raw, _line(node), scope, owner_type, context)
        elif node.type in {"struct_expression", "type_cast_expression"}:
            type_node = _field(node, "name") or _field(node, "type")
            if type_node is not None:
                for reference in self._type_references(type_node):
                    self._add_reference(caller, reference, _line(node), scope, owner_type)

        for child in node.named_children:
            self._walk_body(child, caller, scope, owner_type, context)

    def _add_call(
        self,
        caller: str,
        raw: str,
        line: int,
        scope: tuple[str, ...],
        owner_type: str | None,
        context: dict[str, str],
    ) -> None:
        value = raw.strip().rstrip("!")
        value = re.sub(r"::<.*>$", "", value)
        value = value.replace(" ", "")
        if not value:
            return
        if "." in value:
            root, rest = value.split(".", 1)
            if root in {"self", "Self"} and owner_type:
                value = f"{owner_type}.{rest}"
            elif root in context:
                value = f"{context[root]}.{rest}"
        reference = self._normalize_reference(value, scope, owner_type)
        resolved = self._resolve_reference(reference, scope, owner_type)
        self._append_relationship(caller, resolved or reference, line, resolved is not None)

    def _add_reference(
        self,
        caller: str,
        raw: str,
        line: int,
        scope: tuple[str, ...],
        owner_type: str | None,
    ) -> None:
        reference = self._normalize_reference(raw, scope, owner_type)
        base = self._base_type(reference)
        if base in RUST_PRIMITIVE_TYPES or base in {"Self", "self"}:
            if base in {"Self", "self"} and owner_type:
                resolved = self._resolve_reference(owner_type, scope, owner_type)
                if resolved is not None:
                    self._append_relationship(caller, resolved, line, True)
            return
        resolved = self._resolve_reference(reference, scope, owner_type)
        self._append_relationship(caller, resolved or reference, line, resolved is not None)

    def _normalize_path(self, value: str) -> str:
        return re.sub(r"\s+", "", value.strip()).replace("::", ".")

    def _normalize_reference(
        self,
        value: str,
        scope: tuple[str, ...],
        owner_type: str | None,
    ) -> str:
        value = self._normalize_path(value)
        value = value.lstrip("&* ")
        value = re.sub(r"<.*>$", "", value)
        if value.startswith("self."):
            module = ("crate", *self.file_module, *scope)
            return ".".join((*module, value[5:]))
        if value == "self":
            return ".".join(("crate", *self.file_module, *scope))
        if value.startswith("super."):
            module = ("crate", *self.file_module, *scope[:-1])
            return ".".join((*module, value[6:]))
        if value == "super":
            return ".".join(("crate", *self.file_module, *scope[:-1]))
        if value.startswith("Self.") and owner_type:
            return f"{owner_type}.{value[5:]}"
        root, _, rest = value.partition(".")
        if root in self.imports:
            imported = self.imports[root]
            return f"{imported}.{rest}" if rest else imported
        return value

    def _base_type(self, value: str) -> str:
        value = value.strip().lstrip("&*")
        value = re.sub(r"<.*>$", "", value)
        value = value.replace("::", ".")
        return value.rsplit(".", 1)[-1]

    def _resolve_reference(
        self,
        reference: str,
        scope: tuple[str, ...],
        owner_type: str | None,
    ) -> str | None:
        value = self._normalize_path(reference)
        candidates = [value]
        if value.startswith("crate."):
            prefix = ".".join(("crate", *self.file_module))
            if prefix and value.startswith(prefix + "."):
                candidates.append(value[len(prefix) + 1 :])
            elif value.startswith("crate."):
                candidates.append(value[6:])
        if "." not in value:
            for size in range(len(scope), -1, -1):
                prefix = ".".join(scope[:size])
                candidates.append(f"{prefix}.{value}" if prefix else value)
            if owner_type:
                candidates.append(f"{owner_type}.{value}")
        for candidate in candidates:
            node = self.symbols.get(candidate) or self.qualified_symbols.get(
                self._qualified_name(candidate)
            )
            if node is not None:
                return node.id
        return None

    def _append_relationship(self, caller: str, callee: str, line: int, resolved: bool) -> None:
        key = (caller, callee)
        if key in self._relationship_keys:
            return
        self._relationship_keys.add(key)
        self.call_relationships.append(
            CallRelationship(caller=caller, callee=callee, call_line=line, is_resolved=resolved)
        )


def analyze_rust_file(
    file_path: str, content: str, repo_path: str | None = None
) -> tuple[list[Node], list[CallRelationship]]:
    """Analyze one Rust file and return documentable nodes and relationships."""

    analyzer = TreeSitterRustAnalyzer(file_path, content, repo_path)
    return analyzer.nodes, analyzer.call_relationships
