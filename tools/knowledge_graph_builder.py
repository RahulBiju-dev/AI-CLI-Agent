"""Build and inspect small semantic knowledge graphs without external services."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from typing import Any


CAUSAL_RELATIONS = {"causes", "enables", "increases", "decreases", "prevents", "mitigates"}
NEGATIVE_RELATIONS = {"decreases", "prevents", "mitigates", "contradicts", "inhibits"}
MAX_INFERRED_PATHS = 250
MAX_TRAVERSAL_STATES = 20_000
MAX_IDENTIFIER_CHARS = 200
MAX_LABEL_CHARS = 2_000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def knowledge_graph_builder(
    concepts: list[dict],
    relationships: list[dict],
    query: dict | None = None,
    max_depth: int = 4,
) -> str:
    """Validate a semantic graph and infer explainable multi-hop connections.

    Every inferred connection includes the exact relationship path that supports it;
    the function never invents edges from concept names alone.
    """
    if not isinstance(concepts, list) or not isinstance(relationships, list):
        return _json({"error": "concepts and relationships must be arrays"})
    if query is not None and not isinstance(query, dict):
        return _json({"error": "query must be an object"})
    if len(concepts) > 500 or len(relationships) > 3000:
        return _json({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})

    nodes: dict[str, dict] = {}
    errors: list[str] = []
    for index, concept in enumerate(concepts):
        if not isinstance(concept, dict):
            errors.append(f"concepts[{index}] must be an object")
            continue
        node_id = str(concept.get("id", "")).strip()
        if not node_id:
            errors.append(f"concepts[{index}] is missing id")
        elif len(node_id) > MAX_IDENTIFIER_CHARS or any(ord(char) < 32 for char in node_id):
            errors.append(f"concepts[{index}] id is invalid or exceeds {MAX_IDENTIFIER_CHARS} characters")
        elif node_id in nodes:
            errors.append(f"Duplicate concept id: {node_id}")
        else:
            nodes[node_id] = {
                "id": node_id,
                "label": str(concept.get("label") or node_id)[:MAX_LABEL_CHARS],
                "attributes": concept.get("attributes", {}),
            }

    edges: list[dict] = []
    edge_ids: set[str] = set()
    adjacency: dict[str, list[dict]] = defaultdict(list)
    reverse: dict[str, list[dict]] = defaultdict(list)
    signatures: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, relation in enumerate(relationships):
        if not isinstance(relation, dict):
            errors.append(f"relationships[{index}] must be an object")
            continue
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        relation_type = str(relation.get("type", "related_to")).strip().lower()
        edge_id = str(relation.get("id") or f"r{index + 1}").strip()
        if source not in nodes or target not in nodes:
            errors.append(f"relationships[{index}] references an unknown concept")
            continue
        if not relation_type:
            errors.append(f"relationships[{index}] type cannot be empty")
            continue
        if not edge_id:
            errors.append(f"relationships[{index}] id cannot be empty")
            continue
        if len(edge_id) > MAX_IDENTIFIER_CHARS or any(ord(char) < 32 for char in edge_id):
            errors.append(f"relationships[{index}] id is invalid or exceeds {MAX_IDENTIFIER_CHARS} characters")
            continue
        if edge_id in edge_ids:
            errors.append(f"Duplicate relationship id: {edge_id}")
            continue
        try:
            weight = float(relation.get("weight", 1.0))
        except (TypeError, ValueError, OverflowError):
            errors.append(f"relationships[{index}] weight must be numeric")
            continue
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            errors.append(f"relationships[{index}] weight must be finite and between 0 and 1")
            continue
        edge_ids.add(edge_id)
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": relation_type,
            "weight": weight,
            "evidence": relation.get("evidence", []),
        }
        edges.append(edge)
        adjacency[source].append(edge)
        reverse[target].append(edge)
        signatures[(source, target)].add(relation_type)

    if errors:
        return _json({"error": "Invalid graph", "details": errors})

    try:
        depth_limit = max(1, min(int(max_depth), 8))
    except (TypeError, ValueError, OverflowError):
        return _json({"error": "max_depth must be an integer"})
    start = str((query or {}).get("source", "")).strip() or None
    goal = str((query or {}).get("target", "")).strip() or None
    raw_types = (query or {}).get("relation_types")
    if raw_types is not None and not isinstance(raw_types, list):
        return _json({"error": "query.relation_types must be an array"})
    if raw_types is not None and len(raw_types) > 100:
        return _json({"error": "query.relation_types exceeds the 100-item limit"})
    allowed_types = {
        str(value).strip().lower()
        for value in (raw_types or CAUSAL_RELATIONS)
        if str(value).strip()
    }
    if not allowed_types:
        return _json({"error": "query.relation_types cannot be empty"})
    if start and start not in nodes:
        return _json({"error": f"Unknown query source: {start}"})
    if goal and goal not in nodes:
        return _json({"error": f"Unknown query target: {goal}"})

    paths: list[dict] = []
    traversal_states = 0
    traversal_truncated = False
    origins = [start] if start else list(nodes)
    for origin in origins:
        queue = deque([(origin, [], {origin})])
        while queue and len(paths) < MAX_INFERRED_PATHS and traversal_states < MAX_TRAVERSAL_STATES:
            current, path, visited = queue.popleft()
            traversal_states += 1
            if len(path) >= depth_limit:
                continue
            for edge in adjacency[current]:
                if edge["type"] not in allowed_types or edge["target"] in visited:
                    continue
                new_path = path + [edge]
                destination = edge["target"]
                if len(new_path) >= 2 and (goal is None or destination == goal):
                    sign = -1 if sum(e["type"] in NEGATIVE_RELATIONS for e in new_path) % 2 else 1
                    paths.append({
                        "source": origin,
                        "target": destination,
                        "inferred_effect": "negative" if sign < 0 else "positive",
                        "confidence": round(min(e["weight"] for e in new_path), 4),
                        "path": [{"edge": e["id"], "from": e["source"], "type": e["type"], "to": e["target"]} for e in new_path],
                    })
                queue.append((destination, new_path, visited | {destination}))
        if queue and (
            traversal_states >= MAX_TRAVERSAL_STATES
            or len(paths) >= MAX_INFERRED_PATHS
        ):
            traversal_truncated = True
        if len(paths) >= MAX_INFERRED_PATHS or traversal_states >= MAX_TRAVERSAL_STATES:
            break

    contradictions = []
    for (source, target), types in signatures.items():
        positive = types - NEGATIVE_RELATIONS
        negative = types & NEGATIVE_RELATIONS
        if positive and negative:
            contradictions.append({"source": source, "target": target, "positive": sorted(positive), "negative": sorted(negative)})

    cycles: list[list[str]] = []
    active: list[str] = []
    completed: set[str] = set()

    def find_cycles(node_id: str) -> None:
        if node_id in active:
            cycle = active[active.index(node_id):] + [node_id]
            signature = min(tuple(cycle[index:-1] + cycle[:index] + [cycle[index]]) for index in range(len(cycle) - 1))
            if list(signature) not in cycles:
                cycles.append(list(signature))
            return
        if node_id in completed:
            return
        active.append(node_id)
        for outgoing in adjacency[node_id]:
            find_cycles(outgoing["target"])
        active.pop()
        completed.add(node_id)

    for node_id in nodes:
        find_cycles(node_id)

    centrality = sorted(
        ({"concept": node_id, "degree": len(adjacency[node_id]) + len(reverse[node_id])} for node_id in nodes),
        key=lambda item: (-item["degree"], item["concept"]),
    )
    return _json({
        "graph": {"concepts": list(nodes.values()), "relationships": edges},
        "analysis": {
            "inferred_paths": paths,
            "traversal_truncated": traversal_truncated,
            "contradictions": contradictions,
            "potential_feedback_cycles": cycles,
            "central_concepts": centrality[:10],
        },
        "limits": {
            "max_depth": depth_limit,
            "path_limit": MAX_INFERRED_PATHS,
            "traversal_state_limit": MAX_TRAVERSAL_STATES,
            "traversal_states": traversal_states,
        },
    })
