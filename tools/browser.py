"""
tools/browser.py - Browser control tool.
"""
import urllib.parse
import webbrowser

def open_browser(query: str) -> str:
    """
    Open the default web browser to a specific URL or search query.
    
    Args:
        query (str): The URL to open or a search term.
    
    Returns:
        str: A message indicating success or failure.
    """
    if not query:
        return "Error: No query provided."

    # Check if the query is likely a URL
    if query.startswith("http://") or query.startswith("https://"):
        url = query
    elif "." in query and " " not in query:
        # Simple heuristic for domains like "youtube.com"
        url = f"https://{query}"
    else:
        # It's a search query, use Google search
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
    try:
        success = webbrowser.open(url)
        if success:
            return f"Successfully opened browser with URL: {url}"
        else:
            return f"Failed to open browser with URL: {url}"
    except Exception as e:
        return f"Error opening browser: {str(e)}"
