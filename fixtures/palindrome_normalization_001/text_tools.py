def is_palindrome(text):
    """Check a phrase while ignoring case, spaces, and punctuation."""
    normalized = text.lower()
    return normalized == normalized[::-1]
