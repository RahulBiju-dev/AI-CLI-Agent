# Selene Phase 1 Handoff

## Baseline

- Base commit: `b06c7be5f5f3723cb1471f5d8a0abfb27bf50220`
- Git status at start: clean.
- Reference host inspected: Fedora Linux. `nvidia-smi` was installed but could not return usable GPU data, so automatic selection correctly chose the conservative `low-vram` profile.
- Python: 3.14.6. Ollama CLI: 0.23.2. The local Ollama API/model inventory was inspected without downloading, deleting, rebuilding, or stopping models.
- Initial repository compilation passed. Initial bounded test discovery had one pre-existing collection failure because `tests/test_terminal_launcher.py` imports unavailable `pytest`.
- No private machine output, session contents, credentials, model contents, or runtime data were added to the repository.

## Architecture discovered

- `main.py` owns startup validation and selects the CLI (`agent/core.py`) or threaded HTTP/SSE web runtime (`agent/web.py`).
- Ollama work previously spanned primary chat, title generation, embeddings, vault indexing, and image description. These paths now converge on one local API service and process-wide coordinator.
- The web server had legacy module state shared by clients. Client/session snapshots and active-generation ownership are now isolated in `agent/web_runtime.py`; compatibility globals remain only at the boundary.
- Tool schemas and dispatch originate in `tools/registry.py`. All normal CLI, web, slash-command, pre-index, and routine execution now use `agent/tool_runner.py`.
- Runtime writes cover sessions, aliases, routines, model-build metadata, prompt caches, encrypted Google state, and Chroma-backed indexes. Runtime locations and critical atomic writes are now centralized.
- Native launches and process lifecycle previously mixed platform assumptions across tools. Stable contracts now live in `agent/platform_runtime.py`, with native Linux/Fedora and Windows branches.
- Electron owns exactly the backend child it starts. Ollama remains externally owned and is never stopped by Selene.

## Core design decisions

1. Runtime settings resolve in this order: session/command override, environment or user config, selected hardware profile, conservative defaults. `auto`, `low-vram`, `balanced`, and `manual` are supported.
2. The conservative profile uses a 4096-token context, 768 output-token ceiling, batch size 128, one model slot, one heavy-tool slot, and two tool workers. These are conservative safeguards, not a claim of measured optimality.
3. One FIFO, cancellation-aware, re-entrant Ollama coordinator distinguishes build, chat, title, summary, embedding, and vision work. Low-VRAM mode serializes model-heavy operations without serializing ordinary tools.
4. Selene never starts a duplicate Ollama server or stops an Ollama process. Managed model replacement is built under a unique staging alias, inspected, copied to the live alias, and only then cleaned up. A normalized Modelfile hash records staleness.
5. Every centralized chat and embedding call has a final context guard. CLI/web preparation also accounts for the system prompt, selected tool schemas, history, tool results, continuation prompt, output reserve, images, and safety margin.
6. Tool schemas are compacted and selected per request to keep the first low-VRAM turn viable. Tool continuations retain the assistant call, exact tool results, continuation prompt, and original request as an atomic tail.
7. Saved sessions have one active generation across tabs; unsaved sessions are isolated per browser tab. Generation IDs own cancellation and have one terminal state: `completed`, `cancelled`, or `failed`.
8. Tool metadata is authoritative for side effects, parallel safety, resource weight, cancellation, timeout, output limits, platform support, and optional dependencies. Duplicate results are replayed exactly, output is bounded, temporal preflight is ordered, and uncertain side-effect timeouts block later side effects in that batch.
9. `SELENE_DATA_DIR` remains highest priority. Existing `~/.selene-agent` data is selected in place for compatibility. New Linux locations follow XDG; native Windows uses `%LOCALAPPDATA%\\Selene`. No automatic move or copy occurs.
10. Critical persistence uses same-directory temporary files, flush, optional `fsync`, atomic replace, cleanup, and private permissions where appropriate. Malformed JSON is preserved and surfaced.
11. Packaged resource lookup, backend selection, port readiness, single-instance behavior, owner-token shutdown, pipe draining, and exact child-tree termination are explicit. The PyInstaller backend intentionally retains stdout for the Electron port contract; Windows hides its console when Electron launches it.

Final completion verification also corrected concrete handoff blockers without expanding scope: cross-thread stream closure now always releases its Ollama lease; an Ollama list/show availability race is controlled; fractional integer settings are rejected; relative native runtime-root environment values fall back safely; Windows console fallbacks retain interactive standard handles; routine termination never reports unconfirmed process-tree success; SSE header disconnects release generation ownership; blank-chat start and session deletion share an atomic lifecycle boundary; same-name manual saves are unique; and a tool-round cap cannot persist unmatched tool calls.

## Fedora behavior preserved or improved

- Existing legacy runtime data remains active rather than being silently migrated.
- XDG data, state, config, and cache fallbacks are native and independently resolved.
- Spotify retains the Linux MPRIS DBus backend with a lazy optional import; missing DBus cannot prevent startup.
- Linux application discovery uses bounded desktop-entry locations, including user/system and Flatpak/Snap metadata, and launches trusted records without shell interpolation.
- Terminal selection prefers `kgx`, GNOME Terminal, Konsole, Xfce Terminal, then other explicitly supported native terminals including Ptyxis.
- Browser launch remains native and non-blocking. POSIX process groups are created and terminated only for Selene-owned children.
- Linux-only dependencies remain scoped to Linux instead of being removed for Windows compatibility.

## Windows parity foundations

- Native executable lookup, `%LOCALAPPDATA%` runtime paths, safe detached processes, exact `taskkill /T` ownership, Windows Terminal/PowerShell/Command Prompt selection, Start Menu shortcut discovery, URI/browser opening, and packaged resource lookup are implemented without a Unix compatibility layer.
- Windows launch paths use argument arrays or the native ShellExecute mechanism; raw commands, arbitrary model-supplied executable paths, shells, terminals, and uninstallers are rejected where policy requires.
- Reserved device names and names ending in a space or period are rejected by shared filesystem helpers. Windows containment is drive-aware and case-insensitive.
- Spotify uses an explicit capability-limited URI backend and never claims playback was confirmed.
- `dbus-python` is Linux-scoped. Cross-platform build helpers avoid POSIX shell cleanup and environment syntax.

## Runtime compatibility and migration

- No migration was performed.
- Selection order is explicit `SELENE_DATA_DIR`, an existing legacy store, then the platform-native default.
- Compatibility candidates are reported by the resolver but are never silently moved, copied, deleted, or merged.
- Model-build metadata is new state under the active state directory. A missing record causes one safe managed-model rebuild at a future normal startup; this pass did not perform that live rebuild.
- The shortened Modelfile changes its hash, so the next normal managed-model startup will use the staged rebuild path.

## Files changed

- Core/runtime: `main.py`, `Modelfile`, `agent/cancellation.py`, `agent/core.py`, `agent/model_lifecycle.py`, `agent/ollama_runtime.py`, `agent/persistence.py`, `agent/platform_runtime.py`, `agent/runtime_config.py`, `agent/tool_runner.py`, `agent/web.py`, `agent/web_runtime.py`.
- Web UI: `agent/static/app.js`, `agent/static/index.html`.
- Native tools: `tools/app_launcher.py`, `tools/automated_routine_executor.py`, `tools/browser.py`, `tools/codebase_indexer.py`, `tools/google_workspace.py`, `tools/obsi_vault_writer.py`, `tools/registry.py`, `tools/spotify.py`, `tools/terminal_launcher.py`, `tools/vault_embeddings.py`, `tools/vault_indexer.py`, `tools/vault_search.py`, `tools/vision_describer.py`.
- Packaging/dependencies: `.gitignore`, `electron/main.js`, `electron/build_desktop.py`, `package.json`, `requirements.txt`, `selene-backend.spec`.
- Focused tests: `tests/test_automated_routine_executor.py`, `tests/test_context_budget.py`, `tests/test_main.py`, `tests/test_model_lifecycle.py`, `tests/test_native_desktop_backends.py`, `tests/test_ollama_runtime.py`, `tests/test_persistence.py`, `tests/test_platform_runtime.py`, `tests/test_runtime_config.py`, `tests/test_tool_runner.py`, `tests/test_web_runtime.py`.
- Handoff: `PHASE1_HANDOFF.md`, `PHASE1_HANDOFF.json`.

## Commands and tests run

- Baseline: `git status --short`, `python --version`, `python -m compileall .`, `ollama --version`, `ollama list`, `ollama ps`, Fedora system inspection commands, and bounded existing-test discovery.
- Final focused architecture suite: 119 tests passed with `python -m unittest` across runtime configuration, Ollama coordination/lifecycle, context budgeting, web/session/SSE behavior, tool execution, persistence, platform backends, routines, startup, and existing file/knowledge-graph regressions.
- Earlier full discovery: 110 tests passed and one pre-existing import error remained in `tests/test_terminal_launcher.py` because `pytest` is not installed. Full discovery was not repeated after the focused completion fixes.
- `python -m compileall .`: passed.
- Core/CLI/web/runtime module import smoke: passed without starting a live model operation.
- `bun build agent/static/app.js --outfile=/tmp/selene-app-check.js`: passed.
- Electron-bundled Node syntax checks for `electron/main.js` and `electron/preload.js`: passed.
- `python electron/build_desktop.py --help`: passed as a configuration-level packaging smoke test; no package was built.
- Tool registry validation: passed.
- `git diff --check`: passed.

## Failures and skips

- Pre-existing: `tests/test_terminal_launcher.py` cannot be collected in this environment because it imports unavailable `pytest`. Its behavior is covered by new `unittest` native-backend tests, but the old test was not rewritten in Phase 1.
- Skipped by scope/safety: live chat generation, live staged model rebuild, model download, real GPU benchmark, live Spotify/DBus playback, OAuth, personal vault/index access, full AppImage build, full Windows/NSIS build, signing, and publishing.
- `nvidia-smi` could not communicate usable VRAM data on the inspected host. The intended conservative fallback was selected and tested with mocked 4 GiB detection.

## Known risks

- The safeguards are not a measured performance optimum for the owner's 4 GiB GPU. Phase 2 should record a small real-host smoke result without expanding into a full matrix.
- A Python tool that ignores cooperative cancellation may continue in a bounded daemon worker after a timeout. The runner returns a structured timeout and blocks later side effects in the same batch, but Python cannot safely kill an arbitrary thread.
- Native Chroma, PDF, or third-party calls cannot always be interrupted mid-call; cancellation is checked before/after bounded units and old indexes remain usable until a complete replacement is ready.
- A disconnected SSE client is detected when the next write fails; an Ollama request with no interim chunks remains bounded by its configured request timeout.
- Dynamic tool-schema selection has focused low-context tests, not exhaustive natural-language recall coverage for every registered tool.
- Windows Start Menu discovery is intentionally bounded and filters names/locations, but does not yet resolve and validate every shortcut target. Application launch on Windows remains partial until target validation and broader registered-application coverage receive native integration tests. UNC, locked-file, and long-path behavior also remains unverified.
- Tool `idempotent` metadata is recorded but not yet used to distinguish every exception-after-side-effect uncertainty; timeouts are the currently enforced uncertainty barrier.
- Several legacy `/vault` slash subcommands still reference obsolete dispatch names. Normal registry-backed vault execution is covered, but those compatibility aliases require Phase 2 repair.
- CLI streaming still relies on expected Ollama chunk shapes; malformed third-party response-shape coverage remains Phase 2 work.
- The Electron/PyInstaller contracts were syntax/configuration tested, not exercised as built AppImage or NSIS artifacts. Hidden imports and optional native dependencies require artifact-level smoke tests.
- Direct packaging/publish scripts outside the guarded helper still need Phase 2 normalization so they cannot use a stale backend or bypass `--publish never` during verification workflows.
- Existing Linux artifact naming still contains `Selene-1.0.0` while `package.json` is version 2.2.0; release naming cleanup belongs to Phase 2.

## Exact Phase 2 work

1. Run native Windows 10 and 11 smoke tests for paths, terminals, Start Menu apps (including shortcut-target validation), browser/Spotify capability results, process-tree termination, session persistence, and cancellation.
2. Build and smoke-test one Fedora AppImage and one Windows NSIS artifact, including backend discovery, stdout port announcement, readiness, single-instance behavior, clean owner-token shutdown, and data survival.
3. Complete the repetitive per-tool Fedora/Windows dependency and path audit; add the final truthful support matrix.
4. Add real Windows tests for drive letters, mixed separators, Unicode, reserved names, locked replacements, supported UNC operations, and long paths.
5. Expand tool-schema routing/recall, idempotency/exception uncertainty, and non-cooperative timeout/shutdown tests without weakening side-effect ordering.
6. Resolve the repository's `pytest` collection dependency and run the complete suite in Fedora and Windows CI matrices.
7. Repair legacy `/vault` compatibility aliases and harden malformed Ollama chunk handling in the CLI.
8. Audit packaged hidden imports, direct build/publish scripts, and optional dependency errors, then align artifact naming/version metadata.
9. Complete user documentation, migration notes, CI, support matrices, repetitive compatibility cleanup, and release polish assigned to Phase 2.

## Architecture warnings

- Do not replace Fedora DBus, desktop-entry, XDG, or POSIX process behavior with a lowest-common-denominator generic implementation.
- Do not bypass the shared Ollama coordinator for titles, embeddings, vision, summaries, or tool-owned model calls.
- Do not pre-delete the live Ollama alias; retain staged build, inspection, publication, and metadata ordering.
- Do not remove browser client IDs, generation IDs, terminal-state finalization, or origin-session commit checks.
- Do not reintroduce direct handler calls outside `agent/tool_runner.py`.
- Do not replace atomic critical writes with truncate-in-place JSON writes or silently reset malformed state.
- Do not change the PyInstaller backend to windowed/no-console without replacing Electron's stdout port-announcement contract.
- Do not terminate Ollama or search for processes by name; Selene may terminate only the exact backend and subprocess trees it owns.
