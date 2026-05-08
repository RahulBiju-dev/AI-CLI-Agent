import ollama

long_text = "Here is some text. " * 300  # < 15,000 chars
try:
    response = ollama.chat(
        model='gemma-agent',
        messages=[
            {'role': 'user', 'content': 'Summarize this long document.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'read_document', 'arguments': {'file_path': 'doc.txt'}}}]},
            {'role': 'tool', 'content': long_text}
        ]
    )
    print("RESPONSE:", response['message']['content'])
except Exception as e:
    print("ERROR:", e)

