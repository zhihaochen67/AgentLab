def first_even(values):
    """Return the first even value, or None when one is not present."""
    for value in values:
        if value % 2 == 0:
            return value
        return None

    return None
