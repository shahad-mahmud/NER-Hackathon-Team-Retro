def flatten(data):
    res = []
    for d in data:
        res.extend(d)
    
    return res

def is_digit(text: str) -> bool:
    """Checks if the text contains only Bangla digits. Returns
    True if so, else return false.
    Args:
        text (str): The text to check.
    Returns:
        bool: True if text contains only digits. False otherwise.
    """
    for c in text:
        if not (c in ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯', '.']):
            return False
    return True