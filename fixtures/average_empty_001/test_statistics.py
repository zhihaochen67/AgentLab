from statistics import average


def test_average_of_values():
    assert average([2, 4, 9]) == 5


def test_average_of_empty_values_is_zero():
    assert average([]) == 0.0


def test_average_supports_negative_values():
    assert average([-4, 2]) == -1
