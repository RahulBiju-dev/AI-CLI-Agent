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
