def clamp(value, lower, upper):
    """Constrain value to the inclusive lower and upper bounds."""
    if value < lower:
        result = lower
    elif value > upper:
        result = upper
    else:
        result = value

    return value
