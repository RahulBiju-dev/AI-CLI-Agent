# Gemma CLI Agent

A modular, tool-capable local AI agent built with Python and Ollama. This agent runs directly in your terminal, providing a rich, interactive chat experience with support for web search, document reading, local file manipulation, Spotify playback control, and session persistence.

## Features

- **Local AI execution**: Uses Ollama with the Gemma 4 model (`gemma4:e4b` by default).
- **Tool Calling**: The agent can autonomously decide to use built-in tools:
  - 🔍 **Web Search**: Fetch real-time information using DuckDuckGo (`ddgs`).
  - 📄 **Document Reader**: Extract text from PDF (`pypdf`) and Word documents (`python-docx`).
  - 📂 **File Reader**: Read contents of local text files.
  - 📝 **File Creator**: Create files and write content directly to the local filesystem.
  - 🎵 **Spotify Control**: Open Spotify and play specific songs — by name or Spotify URI/URL.
- **Rich Terminal Interface**: Renders markdown output beautifully in the terminal using the `rich` library.
- **Thought Process Visibility**: Streams the agent's internal reasoning/thinking before providing the final answer.
- **Session Management**: Slash commands allow saving and loading conversations (`/save`, `/load`).
- **Dynamic Configuration**: Adjust parameters (temperature, top_p, etc.), system prompts, and formatting on the fly via `/set` commands.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running locally.
- Python 3.10+
- **For Spotify control**: Spotify desktop app installed (Flatpak, Snap, or native) and `dbus-python` (usually pre-installed on GNOME/Fedora).

## Installation

1. **Clone the repository** (or navigate to your local copy):
   ```bash
   cd Gemma-CLI-Agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: `ollama`, `ddgs`, `rich`, `pypdf`, `python-docx`.*

3. **Ensure Ollama is running:**
   Make sure your Ollama server is running in the background before starting the agent.

## Usage

Start the agent by running:

```bash
python main.py
```

On first run, the app will automatically build the custom model (`gemma-agent`) defined in the `Modelfile`.

### Slash Commands

Inside the chat loop, you can use various commands to control the agent and session:

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

- `main.py`: Entry point that ensures the model is built and launches the agent loop.
- `Modelfile`: Configuration for the base Ollama model, injecting custom system prompts and data cutoff rules.
- `agent/core.py`: Contains the main chat loop, streaming response logic, tool call interception, and command handling.
- `tools/registry.py`: Defines the available tools schema and dispatcher for the agent.
- `tools/search.py`: Implementation of the web search tool using DuckDuckGo.
- `tools/document.py`: PDF and Word document extraction logic.
- `tools/file.py`: Local text file reader and writer functions.
- `tools/spotify.py`: Spotify desktop control via D-Bus (MPRIS2) — launch, search, and play music.

## License

Please refer to the `LICENSE` file in this repository for more information.
