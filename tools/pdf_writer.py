"""Safe PDF creation and exhaustive vault export tools."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from agent.cancellation import CancellationToken, OperationCancelled
from agent.platform_runtime import get_runtime_paths
from agent.persistence import atomic_write_json, atomic_write_text, read_json_preserved

MAX_PDF_CONTENT_CHARS = 10_000_000
EXPORTS_DIR = get_runtime_paths().data_dir / "vaults" / "exports"
PDF_JOBS_DIR = get_runtime_paths().data_dir / "vaults" / ".pdf_jobs"
NOTES_SOURCE_CHARS = 6000


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_output_path(file_path: str) -> Path:
    raw = str(file_path or "").strip()
    if not raw:
        raise ValueError("file_path is required")
    requested = Path(raw).expanduser()
    if requested.suffix.lower() != ".pdf":
        requested = requested.with_suffix(".pdf")
    if requested.is_absolute():
        resolved = requested.resolve()
        allowed_roots = (
            Path.cwd().resolve(),
            get_runtime_paths().data_dir.resolve(),
            EXPORTS_DIR.resolve(),
        )
        if not any(_is_within(resolved, root) for root in allowed_roots):
            raise ValueError(
                "Absolute PDF output must stay inside the current workspace or Selene data directory"
            )
        return resolved
    resolved = (EXPORTS_DIR / requested).resolve()
    if not _is_within(resolved, EXPORTS_DIR.resolve()):
        raise ValueError("Relative PDF output cannot escape Selene's vault exports directory")
    return resolved


def _register_unicode_font(pdfmetrics, TTFont) -> str:
    candidates = (
        Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("SeleneUnicode", str(candidate)))
            return "SeleneUnicode"
        except Exception:
            continue
    return "Helvetica"


def _markdown_story(content: str, title: str, styles, font_name: str):
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Preformatted, Spacer

    body = ParagraphStyle(
        "SeleneBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=13,
        spaceAfter=5,
        wordWrap="CJK",
    )
    heading1 = ParagraphStyle(
        "SeleneHeading1", parent=body, fontSize=17, leading=21, spaceBefore=10,
        spaceAfter=7, textColor="#17365D",
    )
    heading2 = ParagraphStyle(
        "SeleneHeading2", parent=body, fontSize=13, leading=17, spaceBefore=8,
        spaceAfter=5, textColor="#24527A",
    )
    heading3 = ParagraphStyle(
        "SeleneHeading3", parent=body, fontSize=11, leading=14, spaceBefore=6,
        spaceAfter=4, textColor="#2F5D73",
    )
    title_style = ParagraphStyle(
        "SeleneTitle", parent=heading1, fontSize=22, leading=27,
        alignment=TA_CENTER, spaceAfter=12,
    )
    code_style = ParagraphStyle(
        "SeleneCode", parent=body, fontName="Courier", fontSize=8, leading=10,
        leftIndent=4 * mm, backColor="#F3F5F7", borderPadding=5,
    )

    story = [Paragraph(html.escape(title), title_style)] if title else []
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines.clear()
        if text:
            story.append(Paragraph(html.escape(text), body))

    def flush_code() -> None:
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), code_style, maxLineLength=120))
            code_lines.clear()

    for raw_line in content.replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            story.append(Spacer(1, 2 * mm))
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            style = {1: heading1, 2: heading2, 3: heading3}[len(heading.group(1))]
            story.append(Paragraph(html.escape(heading.group(2).strip()), style))
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            story.append(Paragraph(
                f"•&nbsp;&nbsp;{html.escape(bullet.group(1).strip())}",
                ParagraphStyle("SeleneBullet", parent=body, leftIndent=5 * mm, firstLineIndent=-3 * mm),
            ))
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    flush_code()
    if len(story) == (1 if title else 0):
        story.append(Paragraph("No content was provided.", body))
    return story


def _render_pdf(path: Path, title: str, content: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:
        raise RuntimeError("PDF creation requires reportlab. Install project requirements first.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_unicode_font(pdfmetrics, TTFont)
    story = _markdown_story(content, title, getSampleStyleSheet(), font_name)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor("#667085")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    try:
        document = SimpleDocTemplate(
            str(temporary), pagesize=A4,
            rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=17 * mm,
            title=title,
            author="Selene",
        )
        document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_pdf(
    file_path: str,
    content: str | None = None,
    title: str = "",
    content_file: str | None = None,
    overwrite: bool = False,
    confirmed: bool = False,
) -> str:
    """Create a styled PDF atomically from Markdown-like text or a text file."""
    try:
        output = _resolve_output_path(file_path)
        if output.exists() and not overwrite:
            return _json({"error": f"PDF already exists: {output}", "overwrite_required": True})
        if output.exists() and overwrite and not confirmed:
            return _json({"error": "confirmed=true is required to overwrite an existing PDF"})

        if content_file:
            source = Path(content_file).expanduser().resolve()
            if not source.is_file():
                return _json({"error": f"content_file was not found: {source}"})
            if source.stat().st_size > MAX_PDF_CONTENT_CHARS:
                return _json({"error": "content_file exceeds the 10 MB PDF input limit"})
            value = source.read_text(encoding="utf-8")
        else:
            value = str(content or "")
        if not value.strip():
            return _json({"error": "content or content_file is required"})
        if len(value) > MAX_PDF_CONTENT_CHARS:
            return _json({"error": "PDF content exceeds the 10,000,000-character limit"})

        _render_pdf(output, str(title or output.stem).strip(), value)
        return _json({
            "created": True,
            "file_path": str(output),
            "title": str(title or output.stem).strip(),
            "input_characters": len(value),
            "bytes": output.stat().st_size,
        })
    except Exception as exc:
        return _json({"error": str(exc)})


def export_vault_pdf(
    collection: str,
    file_path: str,
    title: str = "",
    source: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    overwrite: bool = False,
    confirmed: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Export every ordered vault chunk to a source-preserving reference PDF."""
    try:
        from tools.vault_indexer import get_chroma_client, resolve_vault_alias
        from tools.vault_search import ordered_vault_records

        collection_name = resolve_vault_alias(collection)
        records = ordered_vault_records(
            collection_name,
            source=source,
            start_page=start_page,
            end_page=end_page,
            cancellation_token=cancellation_token,
        )
        if not records:
            return _json({"error": "No vault chunks matched the requested collection/source/pages"})

        collection_obj = get_chroma_client().get_collection(name=collection_name)
        sections: list[str] = []
        previous_key = None
        previous_end = 0
        for start in range(0, len(records), 100):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            batch = records[start:start + 100]
            raw = collection_obj.get(
                ids=[item["id"] for item in batch],
                include=["documents", "metadatas"],
            )
            by_id = {
                item_id: (document or "", metadata or {})
                for item_id, document, metadata in zip(
                    raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
                )
            }
            for item in batch:
                document, metadata = by_id.get(item["id"], ("", item["metadata"]))
                key = (metadata.get("source"), metadata.get("page"))
                char_start = int(metadata.get("char_start") or 0)
                char_end = int(metadata.get("char_end") or char_start + len(document))
                if key != previous_key:
                    page_label = f" — Page {metadata.get('page')}" if metadata.get("page") else ""
                    sections.append(f"\n## {metadata.get('source', 'Unknown source')}{page_label}\n")
                    previous_end = 0
                    previous_key = key
                overlap = max(0, previous_end - char_start)
                text = str(document)[overlap:]
                if text.strip():
                    sections.append(text.strip())
                previous_end = max(previous_end, char_end)

        content = "\n\n".join(sections)
        if len(content) > MAX_PDF_CONTENT_CHARS:
            return _json({
                "error": "Vault export exceeds the 10,000,000-character PDF input limit",
                "chunks": len(records),
            })
        result = json.loads(create_pdf(
            file_path=file_path,
            content=content,
            title=title or f"{collection_name} Knowledge Export",
            overwrite=overwrite,
            confirmed=confirmed,
        ))
        result.update({
            "collection": collection_name,
            "source": source,
            "exported_chunks": len(records),
            "export_kind": "ordered_source_preserving",
        })
        return _json(result)
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc)})


def _notes_job_paths(collection: str, source: str | None, output: Path) -> tuple[Path, Path]:
    import hashlib

    key = hashlib.sha256(
        f"{collection}:{source or ''}:{output}".encode("utf-8")
    ).hexdigest()[:24]
    directory = PDF_JOBS_DIR / key
    return directory, directory / "state.json"


def _parse_notes_cursor(value: str | int | None) -> tuple[int, int]:
    raw = str(value if value is not None else "0:0").strip()
    try:
        if ":" in raw:
            record, char = raw.split(":", 1)
            return max(0, int(record)), max(0, int(char))
        return max(0, int(raw)), 0
    except ValueError as exc:
        raise ValueError("cursor must be an integer or '<chunk>:<character>'") from exc


def _cursor_text(cursor: tuple[int, int]) -> str:
    return f"{cursor[0]}:{cursor[1]}"


def _vault_source_window(
    collection_obj,
    records: list[dict],
    cursor: tuple[int, int],
    max_chars: int = NOTES_SOURCE_CHARS,
) -> tuple[str, tuple[int, int]]:
    record_index, char_offset = cursor
    candidates = records[record_index:record_index + 20]
    if not candidates:
        return "", cursor
    raw = collection_obj.get(
        ids=[item["id"] for item in candidates],
        include=["documents", "metadatas"],
    )
    by_id = {
        item_id: (str(document or ""), metadata or {})
        for item_id, document, metadata in zip(
            raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
        )
    }
    parts: list[str] = []
    used = 0
    next_cursor = cursor
    previous_key = None
    previous_end = 0
    if record_index > 0 and char_offset == 0:
        previous_meta = records[record_index - 1]["metadata"]
        previous_key = (previous_meta.get("source"), previous_meta.get("page"))
        previous_end = int(previous_meta.get("char_end") or 0)
    for relative_index, item in enumerate(candidates):
        document, metadata = by_id.get(item["id"], ("", item["metadata"]))
        offset = char_offset if relative_index == 0 else 0
        key = (metadata.get("source"), metadata.get("page"))
        char_start = int(metadata.get("char_start") or 0)
        char_end = int(metadata.get("char_end") or char_start + len(document))
        if offset == 0 and key == previous_key:
            offset = min(len(document), max(0, previous_end - char_start))
        if offset >= len(document):
            next_cursor = (record_index + relative_index + 1, 0)
            previous_end = max(previous_end, char_end) if key == previous_key else char_end
            previous_key = key
            continue
        header = (
            f"[Source: {metadata.get('source', 'unknown')} | "
            f"Page: {metadata.get('page', '?')} | "
            f"Chunk: {metadata.get('chunk_index', '?')} | "
            f"Kind: {metadata.get('content_kind', 'text')}]"
        )
        available = max_chars - used - len(header) - 2
        if available <= 0:
            break
        text_slice = document[offset:offset + available]
        parts.append(f"{header}\n{text_slice}")
        used += len(header) + len(text_slice) + 2
        if offset + len(text_slice) < len(document):
            next_cursor = (record_index + relative_index, offset + len(text_slice))
            break
        next_cursor = (record_index + relative_index + 1, 0)
        previous_end = max(previous_end, char_end) if key == previous_key else char_end
        previous_key = key
        if used >= max_chars:
            break
    return "\n\n".join(parts), next_cursor


def _generate_note_section(
    source_text: str,
    title: str,
    cancellation_token: CancellationToken | None,
) -> str:
    from agent.ollama_runtime import OllamaService, OperationKind
    from agent.runtime_config import get_runtime_config

    runtime = get_runtime_config()
    service = OllamaService(runtime)
    owner = f"vault-notes:{threading.get_ident()}:{time.monotonic_ns()}"
    response = service.chat(
        kind=OperationKind.CHAT,
        owner=owner,
        cancellation_token=cancellation_token,
        operation_timeout=runtime.summary_timeout_seconds,
        model=runtime.chat_model,
        stream=False,
        think=False,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the supplied vault excerpt into dense, accurate lecture notes. "
                    "Preserve definitions, equations, steps, tables, diagram relationships, examples, caveats, "
                    "and source/page labels. Remove only repetition and presentation filler. "
                    "Do not add outside facts. Use clear Markdown headings and bullets."
                ),
            },
            {
                "role": "user",
                "content": f"Notes document: {title}\n\nVault excerpt:\n{source_text}",
            },
        ],
    )
    message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not content:
        raise RuntimeError("The local chat model returned an empty notes section")
    return str(content).strip()


def build_vault_notes_pdf(
    collection: str,
    file_path: str,
    title: str = "",
    source: str | None = None,
    cursor: str | int | None = None,
    sections_per_run: int = 4,
    action: str = "build",
    overwrite: bool = False,
    confirmed: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Build grounded notes over an entire vault through resumable model sections."""
    try:
        from tools.vault_indexer import get_chroma_client, resolve_vault_alias
        from tools.vault_search import ordered_vault_records

        action = str(action or "build").strip().lower()
        if action not in {"build", "status"}:
            return _json({"error": "action must be build or status"})
        collection_name = resolve_vault_alias(collection)
        output = _resolve_output_path(file_path)
        job_dir, state_path = _notes_job_paths(collection_name, source, output)
        try:
            state = read_json_preserved(state_path, expected_type=dict)
        except FileNotFoundError:
            state = {}
        if action == "status":
            return _json({
                "collection": collection_name,
                "file_path": str(output),
                "job_directory": str(job_dir),
                "exists": bool(state),
                **state,
            })

        if state.get("complete") and output.is_file():
            return _json({
                "created": True,
                "complete": True,
                "already_complete": True,
                "collection": collection_name,
                "source": source,
                "file_path": str(output),
                "bytes": output.stat().st_size,
                "completed_sections": state.get("completed_sections", 0),
                "job_directory": str(job_dir),
            })
        if state.get("complete") and not output.exists():
            state["complete"] = False
            state["next_cursor"] = f"{state.get('total_chunks', 0)}:0"
            atomic_write_json(state_path, state)

        if output.exists() and not overwrite and not state.get("complete"):
            return _json({"error": f"PDF already exists: {output}", "overwrite_required": True})
        if output.exists() and overwrite and not confirmed:
            return _json({"error": "confirmed=true is required to overwrite an existing PDF"})

        records = ordered_vault_records(
            collection_name,
            source=source,
            cancellation_token=cancellation_token,
        )
        if not records:
            return _json({"error": "No vault chunks matched the requested collection/source"})
        import hashlib
        selection_digest = hashlib.sha256(
            "\n".join(str(item["id"]) for item in records).encode("utf-8")
        ).hexdigest()
        if state and state.get("selection_digest") != selection_digest:
            return _json({
                "error": "The vault changed after this notes job started; use a different file_path to start a fresh grounded export",
                "current_cursor": state.get("next_cursor"),
            })

        if not state:
            job_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "version": 1,
                "collection": collection_name,
                "source": source,
                "file_path": str(output),
                "title": str(title or f"{collection_name} Notes"),
                "selection_digest": selection_digest,
                "total_chunks": len(records),
                "next_cursor": "0:0",
                "completed_sections": 0,
                "section_files": [],
                "complete": False,
            }
            atomic_write_json(state_path, state)

        missing_sections = [
            name for name in state.get("section_files", [])
            if not (job_dir / str(name)).is_file()
        ]
        if missing_sections:
            return _json({
                "error": "A committed notes section is missing; refusing to create an incomplete PDF",
                "missing_sections": missing_sections[:20],
                "job_directory": str(job_dir),
            })

        current_cursor = _parse_notes_cursor(state.get("next_cursor"))
        if cursor is not None and _parse_notes_cursor(cursor) != current_cursor:
            return _json({
                "error": "cursor does not match the durable job checkpoint",
                "next_cursor": _cursor_text(current_cursor),
            })
        sections_per_run = max(1, min(int(sections_per_run), 12))
        collection_obj = get_chroma_client().get_collection(name=collection_name)

        created_this_run = 0
        while current_cursor[0] < len(records) and created_this_run < sections_per_run:
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            source_text, next_cursor = _vault_source_window(
                collection_obj, records, current_cursor
            )
            if not source_text or next_cursor == current_cursor:
                raise RuntimeError(f"Could not advance vault notes cursor {_cursor_text(current_cursor)}")
            section_name = (
                f"section-{current_cursor[0]:08d}-{current_cursor[1]:08d}--"
                f"{next_cursor[0]:08d}-{next_cursor[1]:08d}.md"
            )
            section_path = job_dir / section_name
            if not section_path.exists():
                notes = _generate_note_section(
                    source_text,
                    state["title"],
                    cancellation_token,
                )
                atomic_write_text(section_path, notes, durable=True)
            section_files = list(state.get("section_files", []))
            if section_name not in section_files:
                section_files.append(section_name)
            state["section_files"] = sorted(section_files)
            current_cursor = next_cursor
            created_this_run += 1
            state["next_cursor"] = _cursor_text(current_cursor)
            state["completed_sections"] = len(state["section_files"])
            atomic_write_json(state_path, state)

        if current_cursor[0] >= len(records):
            section_paths = [job_dir / name for name in state.get("section_files", [])]
            notes_content = "\n\n".join(
                path.read_text(encoding="utf-8") for path in section_paths
            )
            result = json.loads(create_pdf(
                file_path=str(output),
                content=notes_content,
                title=state["title"],
                overwrite=overwrite,
                confirmed=confirmed,
            ))
            if result.get("error"):
                result.update({
                    "job_directory": str(job_dir),
                    "sections_preserved": len(section_paths),
                    "next_cursor": _cursor_text(current_cursor),
                })
                return _json(result)
            state["complete"] = True
            state["next_cursor"] = None
            state["pdf_bytes"] = result.get("bytes")
            atomic_write_json(state_path, state)
            result.update({
                "collection": collection_name,
                "source": source,
                "refined": True,
                "completed_sections": len(section_paths),
                "job_directory": str(job_dir),
            })
            return _json(result)

        return _json({
            "collection": collection_name,
            "source": source,
            "file_path": str(output),
            "complete": False,
            "next_cursor": _cursor_text(current_cursor),
            "total_chunks": len(records),
            "completed_sections": state["completed_sections"],
            "sections_created_this_run": created_this_run,
            "job_directory": str(job_dir),
            "guidance": (
                "Call build_vault_notes_pdf again with cursor=next_cursor and the same arguments. "
                "Each call resumes from durable section checkpoints; finalize occurs automatically at the end."
            ),
        })
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc)})
