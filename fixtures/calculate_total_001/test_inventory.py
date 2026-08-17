from inventory import calculate_total


def test_calculate_total():
    assert calculate_total([10, 20, 30]) == 60
