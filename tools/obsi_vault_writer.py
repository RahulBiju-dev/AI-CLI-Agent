"""Vault writer: Creates structured markdown notes optimized for Obsidian Graph View."""

import json
import os
import re
import unicodedata

from agent.platform_runtime import get_runtime_paths

DATA_DIR = str(get_runtime_paths().data_dir)
VAULTS_DIR = os.path.join(DATA_DIR, "vaults")
MAX_NOTE_CHARS = 5_000_000
MAX_TITLE_CHARS = 500
MAX_LINKS = 200
MAX_TAGS = 200
MAX_LINK_CHARS = 500
MAX_TAG_CHARS = 100
MAX_VERSION_ATTEMPTS = 10_000

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _safe_link(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("]]", "] ]").strip()


def _as_list(value: object) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def create_structured_note(
    title: str = "Untitled Note", 
    content: str = "", 
    incoming_links: list[str] | None = None,
    outgoing_links: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Creates a structured Obsidian note optimized for Graph View.
    
    This function generates a markdown file formatted specifically for knowledge
    graph tools like Obsidian. It includes YAML frontmatter for tags, safe filename
    generation to prevent overwrites of existing notes, and automatic appending or
    prepending of cross-reference (wiki) links.
    
    Args:
        title (str): The primary header and base filename for the note.
        content (str): The main body text of the note.
        incoming_links (list[str]): Optional wiki-links to place at the top.
        outgoing_links (list[str]): Optional wiki-links to place in a 'Related Concepts' section.
        tags (list[str]): Optional string tags to place in the YAML frontmatter.
        
    Returns:
        str: A JSON-encoded string indicating success with the created filepath
             or an error message on failure.
    """
    if not title:
        title = "Untitled Note"
    else:
        title = str(title)
    if len(title) > MAX_TITLE_CHARS:
        return _json({"error": f"title exceeds the {MAX_TITLE_CHARS}-character limit"})
    display_title = re.sub(r"[\r\n\x00-\x1f]+", " ", title).strip() or "Untitled Note"
        
    if not content:
        content = ""
    else:
        content = str(content)
    if len(content) > MAX_NOTE_CHARS:
        return _json({"error": f"Note content exceeds the {MAX_NOTE_CHARS}-character limit"})
    for field_name, value, limit in (
        ("incoming_links", incoming_links, MAX_LINKS),
        ("outgoing_links", outgoing_links, MAX_LINKS),
        ("tags", tags, MAX_TAGS),
    ):
        if value is not None and not isinstance(value, list):
            return _json({"error": f"{field_name} must be an array"})
        if isinstance(value, list) and len(value) > limit:
            return _json({"error": f"{field_name} exceeds the {limit}-item limit"})

    try:
        os.makedirs(VAULTS_DIR, exist_ok=True)
    except OSError as exc:
        return _json({"error": f"Failed to prepare the vault directory: {exc}"})
    
    # 2. Graph View Optimization: Sanitize file names (replace invalid characters with dashes)
    safe_title = unicodedata.normalize("NFKC", display_title)
    safe_title = re.sub(r'[\\/*?:"<>|\x00-\x1f]', '-', safe_title)
    safe_title = re.sub(r'\s+', ' ', safe_title).strip()
    safe_title = safe_title.strip(". ")[:180] or "Untitled Note"
    if safe_title.casefold() in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
        safe_title = f"_{safe_title}"
    
    # 4. Prevent Destructive Overwrites: versioned filename
    base_filename = f"{safe_title}.md"
    filepath = os.path.join(VAULTS_DIR, base_filename)
        
    note_lines = []
    
    # Format tags into standard YAML frontmatter
    tags = _as_list(tags)
    incoming_links = _as_list(incoming_links)
    outgoing_links = _as_list(outgoing_links)
    if any(len(str(value)) > MAX_LINK_CHARS for value in [*incoming_links, *outgoing_links]):
        return _json({"error": f"Wiki links may contain at most {MAX_LINK_CHARS} characters"})
    if any(len(str(value)) > MAX_TAG_CHARS for value in tags):
        return _json({"error": f"Tags may contain at most {MAX_TAG_CHARS} characters"})

    if tags:
        note_lines.append("---")
        note_lines.append("tags:")
        clean_tags = (str(tag).strip().strip('#') for tag in tags)
        for tag in dict.fromkeys(tag for tag in clean_tags if tag):
            note_lines.append(f"  - {json.dumps(tag, ensure_ascii=False)}")
        note_lines.append("---")
        note_lines.append("")
        
    note_lines.append(f"# {display_title}")
    note_lines.append("")
    
    # 3. Internal Linking: Append or prepend cross-reference links
    if incoming_links:
        links = [_safe_link(link) for link in incoming_links]
        note_lines.append("**Incoming Links:** " + ", ".join(f"[[{link}]]" for link in links if link))
        note_lines.append("")
        
    note_lines.append(content)
    note_lines.append("")
    
    # If outgoing_links are provided, format into Related Concepts section at the bottom
    if outgoing_links:
        note_lines.append("## Related Concepts")
        for link in outgoing_links:
            clean_link = _safe_link(link)
            if clean_link:
                note_lines.append(f"- [[{clean_link}]]")
        note_lines.append("")
        
    final_content = "\n".join(note_lines)
    
    try:
        counter = 2
        while True:
            try:
                with open(filepath, "x", encoding="utf-8") as f:
                    f.write(final_content)
                break
            except FileExistsError:
                if counter > MAX_VERSION_ATTEMPTS:
                    return _json({"error": "Could not allocate a unique note filename within the version limit"})
                filepath = os.path.join(VAULTS_DIR, f"{safe_title} v{counter}.md")
                counter += 1
    except Exception as e:
        return _json({"error": f"Failed to write note: {str(e)}"})
        
    return _json({
        "status": "success",
        "filepath": filepath,
        "title": title,
        "message": f"Structured note '{os.path.basename(filepath)}' created successfully."
    })
