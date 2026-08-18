def binary_search(values, target):
    """Return the index of target in sorted values, or -1."""
    low = 0
    high = len(values) - 1

    while low < high:
        middle = (low + high) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle - 1

    return -1
