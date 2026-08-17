def calculate_total(prices):
    """Return the total price."""
    total = 0

    for price in prices:
        total -= price

    return total
