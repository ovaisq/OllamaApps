#!/usr/bin/env python3
# ©2025, Ovais Quraishi
"""duckduckgo websearch functions for LLM function calling
"""

from datetime import datetime
from duckduckgo_search import DDGS

def get_web_results_as_html(dicts):
    """Generate an HTML ordered list from a list of dictionaries containing keyphrases.

        This function takes a list of dictionaries where each dictionary is expected to have 
        a 'keyphrase' key. It searches for the given keyphrase using DuckDuckGo and generates 
        an HTML representation in the form of an ordered list, with each item linking to
        search results.


        Notes:
            - The DuckDuckGo search is performed with specific parameters: region set to "us-en",
            safesearch turned off, time limit to a day ('d'), and maximum results limited to 2.
            - If the 'keyphrase' key is missing in a dictionary or its value is empty,
            that dictionary is ignored.
            - Each search result is expected to contain at least a 'title' and an 'href'. 
            If these keys are missing, default values ('No Title', '#') are used.
    """

    ddgs = DDGS()
    html_output = "<ol>\n"

    for d in dicts:
        # Extract the keyphrase from the current dictionary
        keyphrase_orig = d.get('keyphrase', '')
        keyphrase = keyphrase_orig.lower() # only way I could get DuckDuckGo not to 202

        if not keyphrase:
            continue

        # Perform a search with DuckDuckGo for each keyphrase
        try:
            search_results = ddgs.text(keyphrase.lower(),
                                       region="us-en",
                                       safesearch="off",
                                       timelimit="d",
                                       max_results=2,
                                       backend="lite")
            news_results = ddgs.news(keyphrase.lower(),
                                       region="us-en",
                                       safesearch="off",
                                       timelimit="d",
                                       max_results=2)

            # Append the results as links in an HTML ordered list
            html_output += f"  <li><strong>Web Search Results for '{keyphrase_orig}':</strong></li>\n"
            for result in search_results:
                title = result.get('title', 'No Title')
                href = result.get('href', '#')

                # Format each result as an HTML list item containing a link
                html_output += f"  <ul><li><a href='{href}'>{title}</a></li></ul>\n"
            html_output += f"  <li><strong>News Results for '{keyphrase_orig}':</strong></li>\n"
            for result in news_results:
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                iso_string = result.get('date', 'No Date')
                dt_object = datetime.fromisoformat(iso_string)
                news_date = dt_object.strftime("%m-%d-%Y %H:%M")

                # Format each result as an HTML list item containing a link
                html_output += f"  <ul><li><a href='{url}'>{title}</a>&nbsp;<i>({news_date})</i></li></ul>\n"

        except Exception as e:
            # Handle exceptions by appending an error message
            html_output += f"  <li>Error retrieving results for '{keyphrase_orig}': {str(e)}</li>\n"

    html_output += "</ol>"

    return html_output

def create_dict_list_from_text(text):
    """Create a list of dictionaries from structured text input.

        This function processes a given block of text, where each line is expected to start with 
        a number followed by a period (e.g., "1. Example Keyphrase"), and extracts the keyphrases 
        from these lines to create a list of dictionaries. Each dictionary in the returned list 
        contains a single key 'keyphrase' with its corresponding value from the text.

        Notes:
            - The function is case-sensitive and expects each line to start with a number from 1-9, 
            followed by a period.
            - Lines that do not conform to the expected format (e.g., missing numbers or periods)
                are ignored.
            - Leading and trailing whitespace around keyphrases is removed during processing.
    """

    # Initialize an empty list to store dictionaries
    dict_list = []

    # Split the input text into lines
    lines = text.strip().split('\n')

    for line in lines:
        # Check if the line starts with a number followed by a period (e.g., "1.")
        if line.startswith(tuple(f"{i}." for i in range(1, 10))):
            # Extract the keyphrase part of the line
            keyphrase = line.split('.', 1)[1].strip()  # Split at first '.' and take second part
            # Create a dictionary with 'keyphrases' as key
            dict_item = {'keyphrase': keyphrase}
            # Append the dictionary to the list
            dict_list.append(dict_item)

    return dict_list
