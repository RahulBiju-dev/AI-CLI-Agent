# Selene Platform Support Matrix

Generated from the live tool registry (`tools/registry.py`). Fedora Linux is the reference platform; Windows is native parity without WSL.

## Runtime foundations

| Capability | Fedora | Windows | Notes |
|------------|--------|---------|-------|
| Runtime data paths | XDG + legacy `~/.selene-agent` | `%LOCALAPPDATA%\\Selene` + legacy | `SELENE_DATA_DIR` wins; no silent migration |
| Process ownership | POSIX process groups | Detached + `taskkill /T` owned trees | Never kills external Ollama |
| Terminal open | kgx → GNOME Terminal → Konsole → Xfce → Ptyxis | Windows Terminal → pwsh → powershell → cmd | Opens directory only; no command injection |
| App launch | Desktop entries (user/system/Flatpak/Snap) | Bounded Start Menu `.lnk` + target validation | Shells/terminals/uninstallers blocked |
| Browser | Default browser / gio | Default browser / ShellExecute | HTTP(S) only; no credential URLs |
| Spotify | MPRIS DBus (`dbus-python`) | URI handler (capability-limited) | Windows never claims confirmed playback |
| Packaging | AppImage + PyInstaller | NSIS + PyInstaller | Backend stdout port contract retained |

## Registered tools

| Tool | Module | Fedora | Windows | Model-exposed | Optional deps | Side effects | Timeout (s) | Output bound |
|------|--------|--------|---------|---------------|---------------|--------------|-------------|--------------|
| `api_orchestrator` | `tools.api_orchestrator` | supported | supported | yes | — | side-effecting | 180 | 30000 |
| `automated_routine_executor` | `tools.automated_routine_executor` | supported | supported | yes | — | side-effecting | 300 | 20000 |
| `codebase_indexer` | `tools.codebase_indexer` | supported | supported | yes | chromadb, ollama | side-effecting | 900 | 25000 |
| `context_memory_optimizer` | `tools.context_memory_optimizer` | supported | supported | yes | — | read-only | 30 | 25000 |
| `create_file` | `tools.file` | supported | supported | yes | — | side-effecting | 120 | 12000 |
| `create_pdf` | `tools.pdf_writer` | supported | supported | yes | reportlab | side-effecting | 180 | 8000 |
| `create_structured_note` | `tools.obsi_vault_writer` | supported | supported | yes | — | side-effecting | 60 | 12000 |
| `delete_vault_item` | `tools.vault_indexer` | supported | supported | yes | chromadb | side-effecting | 120 | 12000 |
| `describe_image` | `tools.vision_describer` | supported | supported | yes | ollama | read-only | 300 | 24000 |
| `export_vault_pdf` | `tools.pdf_writer` | supported | supported | yes | chromadb, reportlab | side-effecting | 900 | 10000 |
| `get_current_datetime` | `tools.current_datetime` | supported | supported | yes | — | read-only | 5 | 4000 |
| `google_workspace` | `tools.google_workspace` | supported | supported | yes | google-api-python-client, cryptography | side-effecting | 120 | 25000 |
| `index_vault` | `tools.vault_indexer` | supported | supported | yes | chromadb, ollama | side-effecting | 900 | 20000 |
| `build_vault_notes_pdf` | `tools.pdf_writer` | supported | supported | yes | chromadb, ollama, reportlab | side-effecting | 900 | 12000 |
| `knowledge_graph_builder` | `tools.knowledge_graph_builder` | supported | supported | yes | — | read-only | 60 | 30000 |
| `launch_apps` | `tools.app_launcher` | supported | supported | yes | — | side-effecting | 45 | 12000 |
| `list_vault_aliases` | `tools.vault_indexer` | supported | supported | yes | — | read-only | 10 | 12000 |
| `list_vaults` | `tools.vault_indexer` | supported | supported | yes | chromadb | read-only | 60 | 12000 |
| `open_app` | `tools.app_launcher` | supported | supported | no | — | side-effecting | 20 | 8000 |
| `open_browser` | `tools.browser` | supported | supported | yes | — | side-effecting | 15 | 8000 |
| `open_terminal_at_path` | `tools.terminal_launcher` | supported | supported | yes | — | side-effecting | 15 | 8000 |
| `read_document` | `tools.document` | supported | supported | yes | pypdf, python-docx | read-only | 120 | 20000 |
| `read_file` | `tools.file` | supported | supported | yes | — | read-only | 20 | 20000 |
| `reasoning_chain_debugger` | `tools.reasoning_chain_debugger` | supported | supported | yes | — | read-only | 30 | 25000 |
| `register_vault_alias` | `tools.vault_indexer` | supported | supported | no | — | side-effecting | 15 | 8000 |
| `rename_vault` | `tools.vault_indexer` | supported | supported | no | chromadb | side-effecting | 180 | 12000 |
| `run_simulation` | `tools.run_simulation` | supported | supported | yes | — | read-only | 120 | 25000 |
| `spotify_play` | `tools.spotify` | supported | partial | yes | dbus-python | side-effecting | 30 | 8000 |
| `spreadsheet` | `tools.spreadsheet` | supported | supported | yes | openpyxl | side-effecting | 120 | 30000 |
| `vault_search` | `tools.vault_search` | supported | supported | yes | chromadb, ollama | read-only | 180 | 20000 |
| `vault_read` | `tools.vault_search` | supported | supported | yes | chromadb | read-only | 120 | 4000 |
| `view_code` | `tools.code` | supported | supported | yes | — | read-only | 20 | 20000 |
| `web_scrape` | `tools.web_scraper` | supported | supported | yes | requests, beautifulsoup4 | read-only | 90 | 50000 |
| `web_search` | `tools.search` | supported | supported | yes | ddgs | read-only | 90 | 35000 |

## Support status legend

- **supported** — Implemented and covered by automated or platform-contract tests.
- **partial** — Works with documented capability limits (for example Windows Spotify URI launch without playback confirmation).
- **unsupported** — Not available on that platform.
- **limited** — Available only under optional hardware/software constraints.

## Manual verification status

| Area | Status |
|------|--------|
| Fedora unit/platform tests | Automated on Linux CI and local Fedora host |
| Windows unit path/process tests | Automated (mocked / pure helpers); native GUI smoke not run in this Phase 2 pass |
| Full AppImage / NSIS artifact install | Configuration and helper smoke only; full installer builds not executed in this pass |
| Live Ollama chat / model rebuild | Not required for CI; doctor checks API when present |
| Live Spotify / OAuth | Not run; mocked capability paths only |

## Diagnostics

```bash
python main.py --doctor
python main.py --doctor --json
```

## Tests

```bash
python -m compileall . -q
python -m unittest discover -s tests -v
```
