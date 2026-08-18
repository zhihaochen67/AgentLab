from limits import clamp


def test_clamp_raises_value_to_lower_bound():
    assert clamp(-5, 0, 10) == 0


def test_clamp_lowers_value_to_upper_bound():
    assert clamp(15, 0, 10) == 10


def test_clamp_keeps_value_inside_bounds():
    assert clamp(6, 0, 10) == 6
