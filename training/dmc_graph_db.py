"""
dmc_graph_db.py -- Labeled Property Graph (LPG) store for the DMC (Dynamic Memory Controller).

A small, dependency-free, in-memory graph database implementing the node/edge schema
below. It's a plain Python object graph with adjacency indices, JSON persistence, and a
Cypher export -- not a client for a real graph server. If/when this needs to scale past
in-memory, `to_cypher()` gives a starting point for loading the same data into Neo4j (or
any other LPG-speaking store) without changing the schema.

------------------------------------------------------------------------------------------
Node schemas
------------------------------------------------------------------------------------------
:Episodic     node_id, timestamp, usage_counter, retention_value, state_vector
              A temporal anchor for one discrete event.
:Semantic     node_id, entity_class, embedding_vector
              A timeless fact, concept, category, or constraint.
:Procedural   rule_id, activation_condition, execution_step, confidence_score
              A distilled, deterministic skill/rule mined from recurring patterns.

------------------------------------------------------------------------------------------
Edge schemas
------------------------------------------------------------------------------------------
:TEMPORAL_LINK  (:Episodic -> :Episodic)     forward_weight, backward_weight
:INVOLVED_IN    (:Semantic <-> :Episodic)    attention_weight, role
:TRIGGERS       (:Procedural -> :Semantic|:Episodic)   execution_probability

Usage
-----
    from dmc_graph_db import MemoryGraph

    g = MemoryGraph()
    n1 = g.add_node("Episodic", {"timestamp": "2026-07-01T00:00:00+00:00",
                                  "usage_counter": 0, "retention_value": 0.8,
                                  "state_vector": [0.1, 0.2]})
    n2 = g.add_node("Semantic", {"entity_class": "location", "embedding_vector": [0.4, 0.1]})
    g.add_edge("INVOLVED_IN", n2, n1, {"attention_weight": 0.9, "role": "object"})

    g.to_json("memory_graph.json")
    g2 = MemoryGraph.from_json("memory_graph.json")
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

__all__ = ["MemoryGraph", "GNode", "GEdge", "GraphSchemaError"]


# ---------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------

NODE_LABELS = {"Episodic", "Semantic", "Procedural"}
EDGE_TYPES = {"TEMPORAL_LINK", "INVOLVED_IN", "TRIGGERS"}

REQUIRED_NODE_PROPERTIES: Dict[str, set] = {
    "Episodic": {"timestamp", "usage_counter", "retention_value", "state_vector"},
    "Semantic": {"entity_class", "embedding_vector"},
    "Procedural": {"activation_condition", "execution_step", "confidence_score"},
}

REQUIRED_EDGE_PROPERTIES: Dict[str, set] = {
    "TEMPORAL_LINK": {"forward_weight", "backward_weight"},
    "INVOLVED_IN": {"attention_weight", "role"},
    "TRIGGERS": {"execution_probability"},
}

# (source_label, target_label) pairs each edge type is allowed to connect. INVOLVED_IN is
# specified as bidirectional (:Semantic <-> :Episodic); both physical directions are
# accepted so callers can write it either way at add_edge() time.
VALID_EDGE_ENDPOINTS: Dict[str, List[Tuple[str, str]]] = {
    "TEMPORAL_LINK": [("Episodic", "Episodic")],
    "INVOLVED_IN": [("Semantic", "Episodic"), ("Episodic", "Semantic")],
    "TRIGGERS": [("Procedural", "Semantic"), ("Procedural", "Episodic")],
}


class GraphSchemaError(ValueError):
    """Raised when a node/edge violates the LPG schema above."""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------------------

@dataclass
class GNode:
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "properties": dict(self.properties)}


@dataclass
class GEdge:
    id: str
    type: str
    source: str
    target: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "source": self.source,
                "target": self.target, "properties": dict(self.properties)}


class MemoryGraph:
    """An in-memory Labeled Property Graph with adjacency indices for O(1) neighbor
    lookups, schema validation on write, JSON persistence, and a Cypher export."""

    def __init__(self):
        self._nodes: Dict[str, GNode] = {}
        self._edges: Dict[str, GEdge] = {}
        self._out: Dict[str, List[str]] = defaultdict(list)  # node_id -> [edge_id, ...]
        self._in: Dict[str, List[str]] = defaultdict(list)

    # ---- writes ----

    def add_node(self, label: str, properties: Optional[dict] = None,
                 node_id: Optional[str] = None) -> str:
        if label not in NODE_LABELS:
            raise GraphSchemaError(f"Unknown node label {label!r}; expected one of {NODE_LABELS}")
        properties = dict(properties or {})
        missing = REQUIRED_NODE_PROPERTIES[label] - properties.keys()
        if missing:
            raise GraphSchemaError(f":{label} node missing required properties: {sorted(missing)}")

        node_id = node_id or properties.get("node_id") or properties.get("rule_id") or _new_id(label.lower())
        properties.setdefault("node_id" if label != "Procedural" else "rule_id", node_id)
        if node_id in self._nodes:
            raise GraphSchemaError(f"Node id {node_id!r} already exists")

        self._nodes[node_id] = GNode(id=node_id, label=label, properties=properties)
        return node_id

    def add_edge(self, edge_type: str, source_id: str, target_id: str,
                 properties: Optional[dict] = None, edge_id: Optional[str] = None) -> str:
        if edge_type not in EDGE_TYPES:
            raise GraphSchemaError(f"Unknown edge type {edge_type!r}; expected one of {EDGE_TYPES}")
        if source_id not in self._nodes:
            raise GraphSchemaError(f"Unknown source node {source_id!r}")
        if target_id not in self._nodes:
            raise GraphSchemaError(f"Unknown target node {target_id!r}")

        src_label = self._nodes[source_id].label
        tgt_label = self._nodes[target_id].label
        if (src_label, tgt_label) not in VALID_EDGE_ENDPOINTS[edge_type]:
            allowed = VALID_EDGE_ENDPOINTS[edge_type]
            raise GraphSchemaError(
                f":{edge_type} does not allow ({src_label} -> {tgt_label}); allowed: {allowed}"
            )

        properties = dict(properties or {})
        missing = REQUIRED_EDGE_PROPERTIES[edge_type] - properties.keys()
        if missing:
            raise GraphSchemaError(f":{edge_type} edge missing required properties: {sorted(missing)}")

        edge_id = edge_id or _new_id("e")
        if edge_id in self._edges:
            raise GraphSchemaError(f"Edge id {edge_id!r} already exists")

        self._edges[edge_id] = GEdge(id=edge_id, type=edge_type, source=source_id,
                                      target=target_id, properties=properties)
        self._out[source_id].append(edge_id)
        self._in[target_id].append(edge_id)
        return edge_id

    def update_node_properties(self, node_id: str, **updates) -> None:
        if node_id not in self._nodes:
            raise GraphSchemaError(f"Unknown node {node_id!r}")
        self._nodes[node_id].properties.update(updates)

    def remove_node(self, node_id: str, cascade: bool = True) -> None:
        """Removes a node. With cascade=True (default) also removes every edge touching
        it; with cascade=False, raises if the node still has edges (safety net against
        silently orphaning edges)."""
        if node_id not in self._nodes:
            return
        touching = list(self._out.get(node_id, [])) + list(self._in.get(node_id, []))
        if touching and not cascade:
            raise GraphSchemaError(f"Node {node_id!r} still has {len(touching)} edge(s); "
                                    "pass cascade=True to remove them too")
        for edge_id in touching:
            self._remove_edge(edge_id)
        del self._nodes[node_id]
        self._out.pop(node_id, None)
        self._in.pop(node_id, None)

    def _remove_edge(self, edge_id: str) -> None:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return
        if edge_id in self._out.get(edge.source, []):
            self._out[edge.source].remove(edge_id)
        if edge_id in self._in.get(edge.target, []):
            self._in[edge.target].remove(edge_id)

    def merge_nodes(self, keep_id: str, remove_id: str,
                     property_merge_fn: Optional[Callable[[dict, dict], dict]] = None) -> str:
        """Consolidation primitive: rewires every edge touching `remove_id` onto
        `keep_id`, merges properties (default: `keep_id`'s properties win on conflict,
        `property_merge_fn(keep_props, remove_props) -> merged_props` to customize),
        then deletes `remove_id`. Both nodes must share a label. Self-loops that would
        result from the rewire (an edge that already connected keep<->remove) are
        dropped rather than turned into a self-loop."""
        if keep_id not in self._nodes or remove_id not in self._nodes:
            raise GraphSchemaError("merge_nodes requires two existing node ids")
        keep, remove = self._nodes[keep_id], self._nodes[remove_id]
        if keep.label != remove.label:
            raise GraphSchemaError(f"Cannot merge nodes of different labels "
                                    f"({keep.label} vs {remove.label})")

        if property_merge_fn is not None:
            keep.properties = property_merge_fn(keep.properties, remove.properties)

        for edge_id in list(self._out.get(remove_id, [])):
            edge = self._edges[edge_id]
            if edge.target == keep_id:
                self._remove_edge(edge_id)  # would become a self-loop
                continue
            edge.source = keep_id
            self._out[keep_id].append(edge_id)
        self._out.pop(remove_id, None)

        for edge_id in list(self._in.get(remove_id, [])):
            edge = self._edges[edge_id]
            if edge.source == keep_id:
                self._remove_edge(edge_id)
                continue
            edge.target = keep_id
            self._in[keep_id].append(edge_id)
        self._in.pop(remove_id, None)

        del self._nodes[remove_id]
        return keep_id

    # ---- reads ----

    def get_node(self, node_id: str) -> Optional[GNode]:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[GEdge]:
        return self._edges.get(edge_id)

    def neighbors(self, node_id: str, edge_type: Optional[str] = None,
                  direction: str = "out") -> List[Tuple[GNode, GEdge]]:
        """direction: 'out' (edges leaving node_id), 'in' (edges arriving), or 'both'."""
        edge_ids: List[str] = []
        if direction in ("out", "both"):
            edge_ids += self._out.get(node_id, [])
        if direction in ("in", "both"):
            edge_ids += self._in.get(node_id, [])
        results = []
        for eid in edge_ids:
            edge = self._edges[eid]
            if edge_type is not None and edge.type != edge_type:
                continue
            other_id = edge.target if edge.source == node_id else edge.source
            other = self._nodes.get(other_id)
            if other is not None:
                results.append((other, edge))
        return results

    def traverse_temporal(self, start_id: str, direction: str = "forward",
                           max_hops: Optional[int] = None) -> List[GNode]:
        """Walks :TEMPORAL_LINK edges from `start_id`. direction='forward' follows
        source->target (chronological order); 'backward' follows target->source."""
        if start_id not in self._nodes:
            raise GraphSchemaError(f"Unknown node {start_id!r}")
        path = [self._nodes[start_id]]
        current = start_id
        visited = {start_id}
        while max_hops is None or len(path) - 1 < max_hops:
            edge_dir = "out" if direction == "forward" else "in"
            hop = None
            for other, edge in self.neighbors(current, edge_type="TEMPORAL_LINK", direction=edge_dir):
                if edge.type == "TEMPORAL_LINK" and other.id not in visited:
                    hop = other
                    break
            if hop is None:
                break
            path.append(hop)
            visited.add(hop.id)
            current = hop.id
        return path

    def find_nodes(self, label: Optional[str] = None,
                    predicate: Optional[Callable[[GNode], bool]] = None) -> List[GNode]:
        results = [n for n in self._nodes.values() if label is None or n.label == label]
        if predicate is not None:
            results = [n for n in results if predicate(n)]
        return results

    def find_edges(self, edge_type: Optional[str] = None,
                    predicate: Optional[Callable[[GEdge], bool]] = None) -> List[GEdge]:
        results = [e for e in self._edges.values() if edge_type is None or e.type == edge_type]
        if predicate is not None:
            results = [e for e in results if predicate(e)]
        return results

    def stats(self) -> dict:
        node_counts = defaultdict(int)
        for n in self._nodes.values():
            node_counts[n.label] += 1
        edge_counts = defaultdict(int)
        for e in self._edges.values():
            edge_counts[e.type] += 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_label": dict(node_counts),
            "edges_by_type": dict(edge_counts),
        }

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        s = self.stats()
        return (f"MemoryGraph(nodes={s['total_nodes']} {s['nodes_by_label']}, "
                f"edges={s['total_edges']} {s['edges_by_type']})")

    # ---- persistence ----

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGraph":
        g = cls()
        for n in data.get("nodes", []):
            g._nodes[n["id"]] = GNode(id=n["id"], label=n["label"], properties=n["properties"])
        for e in data.get("edges", []):
            edge = GEdge(id=e["id"], type=e["type"], source=e["source"],
                         target=e["target"], properties=e["properties"])
            g._edges[edge.id] = edge
            g._out[edge.source].append(edge.id)
            g._in[edge.target].append(edge.id)
        return g

    @classmethod
    def from_json(cls, path: str) -> "MemoryGraph":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ---- export ----

    def to_cypher(self) -> str:
        """Generates Neo4j-compatible Cypher CREATE statements for every node and edge.
        Not executed anywhere in this module -- just a portable export for loading the
        same graph into a real graph server later."""
        def _lit(value: Any) -> str:
            if isinstance(value, str):
                return json.dumps(value)
            if isinstance(value, (int, float, bool)) or value is None:
                return json.dumps(value)
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(_lit(v) for v in value) + "]"
            return json.dumps(str(value))

        def _props(properties: dict) -> str:
            return "{" + ", ".join(f"{k}: {_lit(v)}" for k, v in properties.items()) + "}"

        lines = []
        var_of = {}
        for i, node in enumerate(self._nodes.values()):
            var = f"n{i}"
            var_of[node.id] = var
            lines.append(f"CREATE ({var}:{node.label} {_props(node.properties)})")
        for edge in self._edges.values():
            src_var = var_of[edge.source]
            tgt_var = var_of[edge.target]
            lines.append(f"CREATE ({src_var})-[:{edge.type} {_props(edge.properties)}]->({tgt_var})")
        return "\n".join(lines)
