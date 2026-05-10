# Gemma CLI Agent

A modular, tool-capable local AI agent built with Python and Ollama. This agent runs directly in your terminal, providing a rich, interactive chat experience with support for web search, document reading, local file manipulation, browser control, Spotify playback, and session persistence.

## Features

- **Local AI execution**: Uses Ollama with the **Gemma 4** model (`gemma4:e4b` by default), customized via a local `Modelfile`.
- **Tool Calling**: The agent autonomously decides when to use its built-in toolkit:
  - 🔍 **Web Search**: Fetch real-time information using DuckDuckGo.
  - 🌐 **Browser Control**: Open specific URLs or perform Google searches directly in your system's default browser.
  - 💻 **Code Viewer**: Read source code files with syntax-aware line numbers and scan directories recursively for specific languages.
  - 📄 **Document Reader**: Extract text from PDF (`pypdf`) and Word documents (`python-docx`).
  - 📂 **File Manager**: Read existing files and create new ones directly on your filesystem.
  - 🎵 **Spotify Control**: Search and play songs, **albums**, or **playlists** using D-Bus (Linux) or web URLs.
- **Rich Terminal Interface**:
  - **Clean Output**: Uses `rich.live` for smooth markdown streaming, eliminating terminal flicker and duplicated lines.
  - **Advanced LaTeX Math**: Support for complex mathematical notation, including Greek letters, fractions, roots, superscripts/subscripts, and environments like `\array` and `\hline`.
- **Thought Process Visibility**: Streams the agent's internal reasoning/thinking before providing the final answer (can be toggled).
- **Session Management**: Persistent chat history with `/save` and `/load` commands.
- **Dynamic Configuration**: Adjust model parameters (temperature, top_p), system prompts, and formatting on the fly.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running locally.
- Python 3.10+
- **For Spotify control**: Spotify desktop app installed and `dbus-python` (for Linux/MPRIS control).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RahulBiju-dev/AI-CLI-Agent.git
   cd AI-CLI-Agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: `ollama`, `ddgs`, `rich`, `pypdf`, `python-docx`.*

3. **Ensure Ollama is running:**
   Make sure your Ollama server is running (`ollama serve`) before starting the agent.

## Usage

Start the agent by running:

```bash
python main.py
```

On first run, the app will automatically build the custom model (`gemma-agent`) defined in the `Modelfile`.

### Slash Commands

Inside the chat loop, use these commands to control the experience:

- `/help` — Show available commands.
- `/clear` — Clear current conversation history.
- `/save [name]` — Save the current session to a JSON file.
- `/load [name|index]` — Load a previously saved session.
- `/set parameter <name> <val>` — Tweak model settings (e.g., `/set parameter temperature 0.7`).
- `/set system "<prompt>"` — Update the system prompt dynamically.
- `/set history` / `/set nohistory` — Toggle context memory.
- `/set think` / `/set nothink` — Toggle visibility of model reasoning.
- `/show parameters` — View active session parameters.
- `/show system` — Display the current system prompt.
- `/show model` — View base model information.
- `/quit` (or `/exit`, `/q`) — Exit the application.

## Project Structure

- `main.py`: Entry point and model initialization.
- `Modelfile`: Custom model configuration and system instructions.
- `agent/core.py`: Main logic for chat loop, tool interception, and streaming.
- `agent/terminal.py`: UI helpers and LaTeX rendering engine.
- `tools/registry.py`: Central registry and dispatcher for all agent tools.
- `tools/browser.py`: Browser navigation and search control.
- `tools/code.py`: Source code inspection and directory scanning.
- `tools/search.py`: DuckDuckGo web search integration.
- `tools/document.py`: PDF/Word document parsing.
- `tools/file.py`: Basic file I/O operations.
- `tools/spotify.py`: Spotify playback and media control.

## License

Distributed under the MIT License. See `LICENSE` for more information.
