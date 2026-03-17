#!/usr/bin/env python3
# ©2025, Ovais Quraishi
"""duckduckgo websearch functions for LLM function calling
"""

from datetime import datetime
from duckduckgo_search import DDGS

def get_web_results_as_html(dicts):
    """Generate an HTML ordered list from a list of dictionaries containing keyphrases."""

    ddgs = DDGS()
    html_output = "<ol>\n"

    for d in dicts:
        keyphrase_orig = d.get('keyphrase', '')
        keyphrase = keyphrase_orig.lower()

        if not keyphrase:
            continue

        html_output += f"  <li><strong>Web Search Results for '{keyphrase_orig}':</strong></li>\n"
        html_output += _fetch_and_format_results(ddgs, keyphrase, keyphrase_orig, 'text')
        html_output += f"  <li><strong>News Results for '{keyphrase_orig}':</strong></li>\n"
        html_output += _fetch_and_format_results(ddgs, keyphrase, keyphrase_orig, 'news')

    html_output += "</ol>"
    return html_output


def _fetch_and_format_results(ddgs, keyphrase, keyphrase_orig, search_type):
    """Fetch search results and format them as HTML list items."""
    
    try:
        if search_type == 'text':
            results = ddgs.text(keyphrase, region="us-en", safesearch="off",
                               timelimit="d", max_results=2, backend="lite")
        else:
            results = ddgs.news(keyphrase, region="us-en", safesearch="off",
                               timelimit="d", max_results=2)

        items_html = ""
        for result in results:
            title = result.get('title', 'No Title')
            url = result.get('href', result.get('url', '#'))
            
            if search_type == 'news':
                iso_string = result.get('date', 'No Date')
                try:
                    news_date = datetime.fromisoformat(iso_string).strftime("%m-%d-%Y %H:%M")
                except (ValueError, AttributeError):
                    news_date = 'Unknown date'
                items_html += f"<ul><li><a href='{url}'>{title}</a>&nbsp;<i>({news_date})</i></li></ul>\n"
            else:
                items_html += f"<ul><li><a href='{url}'>{title}</a></li></ul>\n"
        
        return items_html
        
    except Exception as e:
        return f"  <ul><li>Error retrieving results for '{keyphrase_orig}': {str(e)}</li></ul>\n"

def create_dict_list_from_text(text):
    """Create a list of dictionaries from structured text input."""

    dict_list = []
    lines = text.strip().split('\n')

    for line in lines:
        if line.startswith(tuple(f"{i}." for i in range(1, 10))):
            keyphrase = line.split('.', 1)[1].strip()
            dict_item = {'keyphrase': keyphrase}
            dict_list.append(dict_item)

    return dict_list
