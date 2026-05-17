"""Tool for multimodal vision support using Ollama moondream model."""

import os

def describe_image(image_path: str, prompt: str = "Please provide a highly detailed description of this technical diagram, flowchart, or architecture slide.") -> str:
    """Send an image to the local moondream model and return its description."""
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    try:
        import ollama
        response = ollama.chat(
            model="moondream",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error describing image: {str(e)}"
