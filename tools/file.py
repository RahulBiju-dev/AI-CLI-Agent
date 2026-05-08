import json
import os

def read_file(file_path: str) -> str:
    """Read the contents of a text file from the local filesystem.

    Args:
        file_path: The absolute or relative path to the file.

    Returns:
        A JSON string containing the file's text or an error message.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if len(text) > 15000:
            text = text[:15000] + f"\n\n...[File truncated due to size limit. Showing first 15000 characters.]"

        return json.dumps({"text": text})
    except UnicodeDecodeError:
        return json.dumps({"error": f"Cannot read file: {file_path} appears to be a binary file."})
    except Exception as e:
        return json.dumps({"error": f"Error reading file: {str(e)}"})


def create_file(file_path: str, content: str) -> str:
    """Create a new .txt or .md file with the given content.

    Args:
        file_path: The absolute or relative path where the file should be created. Must end in .txt or .md.
        content: The text content to write to the file.

    Returns:
        A JSON string indicating success or failure.
    """
    if not (file_path.endswith(".txt") or file_path.endswith(".md")):
        return json.dumps({"error": "Only .txt and .md files are supported."})

    try:
        # Create directories if they don't exist
        dir_name = os.path.dirname(os.path.abspath(file_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return json.dumps({"success": True, "message": f"Successfully created file at {file_path}"})
    except Exception as e:
        return json.dumps({"error": f"Error creating file: {str(e)}"})
