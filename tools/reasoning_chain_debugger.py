"""Audit an explicit claim/evidence graph without exposing private chain-of-thought."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


MAX_STEPS = 500
MAX_EVIDENCE = 1_000
MAX_REFERENCES_PER_STEP = 200
MAX_IDENTIFIER_CHARS = 200
MAX_CYCLE_ISSUES = 200


def _reference_list(value: Any, field: str, step_id: str, issues: list[dict]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        issues.append({"severity": "error", "step": step_id, "issue": f"{field} must be an array"})
        return []
    if len(value) > MAX_REFERENCES_PER_STEP:
        issues.append({
            "severity": "error",
            "step": step_id,
            "issue": f"{field} exceeds the {MAX_REFERENCES_PER_STEP}-item limit",
        })
        value = value[:MAX_REFERENCES_PER_STEP]
    references: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > MAX_IDENTIFIER_CHARS:
            issues.append({
                "severity": "error",
                "step": step_id,
                "issue": f"{field} entries must be non-empty strings of at most {MAX_IDENTIFIER_CHARS} characters",
            })
            continue
        references.append(item)
    return references


def _finite_score(value: Any, default: float) -> tuple[float, bool]:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if score != score or score in {float("inf"), float("-inf")}:
        return default, False
    return score, True


def _mermaid_label(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())[:limit]
    return (
        text.replace("\\", "/")
        .replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
    )


def reasoning_chain_debugger(
    conclusion: str,
    steps: list[dict],
    evidence: list[dict] | None = None,
) -> str:
    """Find unsupported claims, dependency gaps, cycles, and confidence problems."""
    if not isinstance(conclusion, str):
        return json.dumps({"error": "conclusion must be a string"})
    if len(conclusion) > 10_000:
        return json.dumps({"error": "conclusion exceeds the 10,000-character limit"})
    if not isinstance(steps, list):
        return json.dumps({"error": "steps must be an array"})
    if evidence is not None and not isinstance(evidence, list):
        return json.dumps({"error": "evidence must be an array"})
    if len(steps) > MAX_STEPS or len(evidence or []) > MAX_EVIDENCE:
        return json.dumps({
            "error": f"Reasoning graph exceeds the {MAX_STEPS}-step/{MAX_EVIDENCE}-evidence limit"
        })
    evidence = evidence or []
    step_map: dict[str, dict] = {}
    issues: list[dict] = []
    evidence_map: dict[str, dict] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            issues.append({"severity": "error", "evidence": index, "issue": "Evidence must be an object"})
            continue
        evidence_id = str(item.get("id") or "").strip()
        if not evidence_id:
            issues.append({"severity": "error", "evidence": index, "issue": "Evidence is missing id"})
        elif len(evidence_id) > MAX_IDENTIFIER_CHARS or any(ord(char) < 32 for char in evidence_id):
            issues.append({
                "severity": "error",
                "evidence": index,
                "issue": f"Evidence id must contain at most {MAX_IDENTIFIER_CHARS} printable characters",
            })
        elif evidence_id in evidence_map:
            issues.append({"severity": "error", "evidence": evidence_id, "issue": "Duplicate evidence id"})
        else:
            evidence_map[evidence_id] = item
    evidence_usage: dict[str, int] = defaultdict(int)
    normalized_dependencies: dict[str, list[str]] = {}
    normalized_references: dict[str, list[str]] = {}

    for index, step in enumerate(steps or []):
        if not isinstance(step, dict):
            issues.append({"severity": "error", "step": index, "issue": "Step must be an object"})
            continue
        step_id = str(step.get("id") or f"step-{index + 1}")
        if len(step_id) > MAX_IDENTIFIER_CHARS or any(ord(char) < 32 for char in step_id):
            issues.append({
                "severity": "error",
                "step": index,
                "issue": f"Step id must contain at most {MAX_IDENTIFIER_CHARS} printable characters",
            })
            step_id = f"step-{index + 1}"
        if step_id in step_map:
            issues.append({"severity": "error", "step": step_id, "issue": "Duplicate step id"})
            continue
        step_map[step_id] = {**step, "id": step_id}

    dependencies: dict[str, list[str]] = defaultdict(list)
    for step_id, step in step_map.items():
        claim = str(step.get("claim", "")).strip()
        deps = _reference_list(step.get("depends_on", []), "depends_on", step_id, issues)
        refs = _reference_list(step.get("evidence_ids", []), "evidence_ids", step_id, issues)
        dependencies[step_id] = deps
        normalized_dependencies[step_id] = deps
        normalized_references[step_id] = refs
        if not claim:
            issues.append({"severity": "error", "step": step_id, "issue": "Missing claim"})
        missing_deps = [dep for dep in deps if dep not in step_map]
        if missing_deps:
            issues.append({"severity": "error", "step": step_id, "issue": "Unknown dependencies", "values": missing_deps})
        missing_refs = [ref for ref in refs if ref not in evidence_map]
        if missing_refs:
            issues.append({"severity": "error", "step": step_id, "issue": "Unknown evidence references", "values": missing_refs})
        if not deps and not refs and not step.get("assumption"):
            issues.append({"severity": "warning", "step": step_id, "issue": "Unsupported root claim; add evidence or mark it as an assumption"})
        confidence, confidence_valid = _finite_score(step.get("confidence", 0.5), 0.5)
        if not confidence_valid:
            issues.append({"severity": "warning", "step": step_id, "issue": "Invalid confidence; treated as 0.5"})
        elif not 0.0 <= confidence <= 1.0:
            issues.append({"severity": "warning", "step": step_id, "issue": "Confidence should be between 0 and 1"})
            confidence = max(0.0, min(1.0, confidence))
        if confidence > 0.8 and not refs and not deps:
            issues.append({"severity": "warning", "step": step_id, "issue": "High confidence is not supported by evidence or prior steps"})
        for ref in refs:
            evidence_usage[ref] += 1
            item = evidence_map.get(ref, {})
            quality, _quality_valid = _finite_score(item.get("quality", 1.0), 1.0)
            if confidence > 0.8 and quality < 0.5:
                issues.append({"severity": "warning", "step": step_id, "issue": "Confidence substantially exceeds cited evidence quality", "evidence": ref})

    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_issue_count = 0

    def visit(step_id: str, trail: list[str]) -> None:
        nonlocal cycle_issue_count
        if step_id in visiting:
            cycle_issue_count += 1
            if cycle_issue_count <= MAX_CYCLE_ISSUES:
                issues.append({"severity": "error", "step": step_id, "issue": "Circular dependency", "path": trail + [step_id]})
            return
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in dependencies[step_id]:
            if dependency in step_map:
                visit(dependency, trail + [step_id])
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_map:
        visit(step_id, [])
    if cycle_issue_count > MAX_CYCLE_ISSUES:
        issues.append({
            "severity": "error",
            "step": None,
            "issue": f"Additional circular dependencies were omitted after {MAX_CYCLE_ISSUES} reports",
        })

    conclusion_supported = any(str(step.get("claim", "")).strip().casefold() == conclusion.strip().casefold() for step in step_map.values())
    if conclusion and not conclusion_supported:
        issues.append({"severity": "warning", "step": None, "issue": "No step explicitly establishes the final conclusion"})
    for ref, count in evidence_usage.items():
        if count >= 3 and len(evidence_map) > 1:
            issues.append({"severity": "warning", "step": None, "issue": "A single evidence item carries several claims; check for undue weight", "evidence": ref, "usage_count": count})

    lines = ["flowchart TD"]
    mermaid_ids = {step_id: f"n{index}" for index, step_id in enumerate(step_map, start=1)}
    for step_id, step in step_map.items():
        safe_id = _mermaid_label(step_id, 80)
        safe_claim = _mermaid_label(step.get("claim", ""), 100)
        lines.append(f'  {mermaid_ids[step_id]}["{safe_id}: {safe_claim}"]')
        for dependency in normalized_dependencies[step_id]:
            if dependency in step_map:
                lines.append(f'  {mermaid_ids[dependency]} --> {mermaid_ids[step_id]}')

    return json.dumps({
        "conclusion": conclusion,
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "issues": issues,
        "evidence_coverage": {
            "provided": len(evidence_map),
            "referenced": len({ref for refs in normalized_references.values() for ref in refs}),
        },
        "mermaid": "\n".join(lines),
        "privacy_note": "This audits the explicit rationale supplied to the tool; it does not reveal hidden model chain-of-thought.",
    }, ensure_ascii=False)
