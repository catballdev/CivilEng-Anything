"""
Clause tree navigation for CivilEng-Anything.

Builds a deterministic tree structure from the canonical section_path hierarchy.
Exposed via CLI (civileng tree), HTTP (GET /documents/{id}/tree), and MCP (get_document_structure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import ContentItem, ItemType
from .store import Store


@dataclass
class TreeNode:
    """Node in the document clause tree."""
    item_id: str
    number: Optional[str]
    type: ItemType
    title: str  # First ~80 chars of text as title
    section_path: list[str]
    pdf_page: int
    children: list["TreeNode"] = field(default_factory=list)
    # Summary fields (nullable, structure only)
    summary_en: Optional[str] = None
    summary_tc: Optional[str] = None
    generated_questions: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "item_id": self.item_id,
            "number": self.number,
            "type": self.type.value,
            "title": self.title,
            "section_path": self.section_path,
            "pdf_page": self.pdf_page,
            "summary_en": self.summary_en,
            "summary_tc": self.summary_tc,
            "generated_questions": self.generated_questions,
            "children": [child.to_dict() for child in self.children],
        }


def build_document_tree(
    store: Store,
    document_id: str,
    edition_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Build deterministic clause tree for a document from section_path hierarchy.
    
    The tree is built by:
    1. Fetching all section and clause items for the document/edition
    2. Sorting by section_path (lexicographic gives correct hierarchy order)
    3. Building parent-child relationships from section_path prefixes
    4. Including paragraphs under their parent clause
    
    Returns None if document not found.
    """
    doc = store.get_document(document_id)
    if not doc:
        return None
    
    if edition_id is None:
        edition_id = doc.edition_id
    
    # Get all sections and clauses (the structural items)
    conn = store._get_conn()
    rows = conn.execute(
        """
        SELECT * FROM items
        WHERE document_id = ? AND edition_id = ? AND type IN (?, ?)
        ORDER BY section_path, pdf_page, item_id
        """,
        (document_id, edition_id, ItemType.SECTION.value, ItemType.CLAUSE.value),
    ).fetchall()
    
    if not rows:
        return {
            "document_id": document_id,
            "edition_id": edition_id,
            "title": doc.title,
            "tree": [],
        }
    
    # Convert rows to items
    items = [store._row_to_item(row) for row in rows]
    
    # Build tree from section_path
    # section_path is like ["1"] for section 1, ["1", "2"] for clause 1.2, etc.
    root_nodes: list[TreeNode] = []
    path_to_node: dict[tuple, TreeNode] = {}  # Maps section_path tuple to TreeNode
    
    for item in items:
        path_tuple = tuple(item.section_path)
        
        # Create node
        title = (item.text or "").strip()
        if len(title) > 80:
            title = title[:77] + "..."
        
        node = TreeNode(
            item_id=item.item_id,
            number=item.number,
            type=item.type,
            title=title,
            section_path=item.section_path,
            pdf_page=item.pdf_page,
            summary_en=item.summary_en,
            summary_tc=item.summary_tc,
            generated_questions=item.generated_questions,
        )
        
        # Find parent: section_path without last element
        parent_path = path_tuple[:-1]
        
        if parent_path:
            parent_node = path_to_node.get(parent_path)
            if parent_node:
                parent_node.children.append(node)
            else:
                # Parent not found (shouldn't happen with proper ordering), attach to root
                root_nodes.append(node)
        else:
            # Top-level section
            root_nodes.append(node)
        
        # Register this node for children to find
        path_to_node[path_tuple] = node
    
    # Now add paragraphs as children of their parent clause/section
    para_rows = conn.execute(
        """
        SELECT * FROM items
        WHERE document_id = ? AND edition_id = ? AND type = ?
        ORDER BY section_path, pdf_page, item_id
        """,
        (document_id, edition_id, ItemType.PARAGRAPH.value),
    ).fetchall()
    
    for row in para_rows:
        item = store._row_to_item(row)
        path_tuple = tuple(item.section_path)
        parent_path = path_tuple[:-1]
        
        title = (item.text or "").strip()
        if len(title) > 80:
            title = title[:77] + "..."
        
        node = TreeNode(
            item_id=item.item_id,
            number=item.number,
            type=item.type,
            title=title,
            section_path=item.section_path,
            pdf_page=item.pdf_page,
            summary_en=item.summary_en,
            summary_tc=item.summary_tc,
            generated_questions=item.generated_questions,
        )
        
        parent_node = path_to_node.get(parent_path)
        if parent_node:
            parent_node.children.append(node)
        else:
            # Orphan paragraph, attach to root
            root_nodes.append(node)
    
    return {
        "document_id": document_id,
        "edition_id": edition_id,
        "title": doc.title,
        "tree": [node.to_dict() for node in root_nodes],
    }


def get_tree_as_text(tree_data: dict[str, Any], indent: int = 0) -> str:
    """Render tree as human-readable text for CLI output."""
    lines = []
    lines.append(f"{tree_data['document_id']} ({tree_data['edition_id']}): {tree_data['title']}")
    
    def render_nodes(nodes: list[dict], level: int = 0):
        prefix = "  " * level
        for node in nodes:
            num = f"{node['number']} " if node['number'] else ""
            type_label = f"[{node['type']}]"
            lines.append(f"{prefix}{num}{type_label} {node['title']}")
            if node['children']:
                render_nodes(node['children'], level + 1)
    
    render_nodes(tree_data['tree'])
    return "\n".join(lines)