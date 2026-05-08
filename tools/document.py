"""
tools/document.py — Document parsing tools for PDF and Word files.
"""

import json
import os


def read_document(file_path: str) -> str:
    """Extract and read text from a PDF or Word document (.docx).

    Args:
        file_path: The path to the document file.

    Returns:
        A JSON string containing the extracted text or an error message.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            import pypdf

            text = ""
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                    if len(text) > 15000:
                        break
            
            text = text.strip()
            if len(text) > 15000:
                text = text[:15000] + f"\n\n...[Document truncated due to size limit. Showing first 15000 characters.]"

            return json.dumps({"text": text})

        elif ext == ".docx":
            import docx

            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            text = text.strip()
            if len(text) > 15000:
                text = text[:15000] + f"\n\n...[Document truncated due to size limit. Showing first 15000 characters.]"

            return json.dumps({"text": text})

        else:
            return json.dumps(
                {"error": f"Unsupported file type: {ext}. Only .pdf and .docx are supported."}
            )

    except ImportError as e:
        pkg = "pypdf" if ext == ".pdf" else "python-docx"
        return json.dumps(
            {"error": f"Missing required dependency. Please run: pip install {pkg}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Error reading document: {str(e)}"})
