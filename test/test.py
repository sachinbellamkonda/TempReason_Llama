import re

def parse_events(cot_response):
    """
    Parses temporal events from a Chain of Thought (CoT) model response.

    Args:
        cot_response (str): The raw text output from the model.

    Returns:
        list: A list of tuples in the format (event description, start date, end date).
    """
    # List to store parsed events
    events = []

    # Define regex pattern for extracting tuples
    tuple_pattern = r'\(\s*"(.*?)"\s*,\s*"(.*?)"\s*,\s*"(.*?)"\s*\)'

    # Remove newline artifacts from the input, then search for patterns
    cleaned_response = re.sub(r'\s*\n\s*', ' ', cot_response)  # Handle multiline cases
    matches = re.findall(tuple_pattern, cleaned_response)

    # Append valid matches to events
    for match in matches:
        # Clean each group to ensure no extra whitespace
        event = tuple(field.strip() for field in match)
        events.append(event)

    return events

# Example Usage
if __name__ == "__main__":
    # Example CoT output from Llama 3.1 8b model
    cot_response = """
    [
    ("Clarence Norman Brunsdale's birth", "July 9, 1891", "July 9, 1891"),
    ("Knute H. Brunsdale's death (father of Clarence Norman Brunsdale)", "1899", "1899"),
    ("Clarence Norman Brunsdale graduated from Luther College", "1913", "1913"),
    ("Clarence Norman Brunsdale married Carrie Lajord", "August 30, 1925", "August 30, 1925"),
    ("Clarence Norman Brunsdale served in North Dakota State Senate", "1927", "1934"),
    ("Clarence Norman Brunsdale served again in North Dakota State Senate", "1941", "1951"),
    ("Clarence Norman Brunsdale was an alternate delegate to Republican National Convention", "1940", "1940"),
    ("Clarence Norman Brunsdale was a member of Republican National Committee from North Dakota", "1948", "1952"),
    ("Clarence Norman Brunsdale served as Governor of North Dakota", "1951", "1957"),
    ("Clarence Norman Brunsdale was appointed U.S. Senator", "November 19, 1959", "November 19, 1959"),
    ("Clarence Norman Brunsdale's term as U.S. Senator", "November 19, 1959", "August 7, 1960"),
    ("Clarence Norman Brunsdale's death", "January 27, 1978", "January 27, 1978"),
    ("Clarence Norman Brunsdale was buried in Mayville Cemetery", "1978", "1978")
]
    """

    # Parse the events
    parsed_events = parse_events(cot_response)

    # Output parsed events
    for event in parsed_events:
        print(event)
