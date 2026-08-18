from parity import first_even


def test_first_even_skips_odd_values():
    assert first_even([1, 3, 8, 10]) == 8


def test_first_even_can_be_first_value():
    assert first_even([4, 7, 8]) == 4


def test_first_even_returns_none_when_missing():
    assert first_even([1, 3, 5]) is None


def test_first_even_handles_empty_input():
    assert first_even([]) is None
