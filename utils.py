import re

def categorize_size(string):
    # Search for patterns like '1B', '1.5B', '3B', '7B', or '8B' (case-insensitive)
    match = re.search(r'(\d+(\.\d+)?)[Bb]', string)
    
    if match:
        # Extract the numeric part and convert it to float for comparison
        size = float(match.group(1))
        
        # Map to categories based on the value of `size`
        if size in [7, 8]:
            return "7B"
        elif size == 3:
            return "3B"
        elif size in [1, 1.5]:
            return "1B"
    
    # Return None or a default category if no match is found
    return None