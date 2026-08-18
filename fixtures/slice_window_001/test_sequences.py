from sequences import take_first


def test_take_first_returns_requested_prefix():
    assert take_first(["a", "b", "c", "d"], 3) == ["a", "b", "c"]


def test_take_first_handles_one_item():
    assert take_first(["a", "b"], 1) == ["a"]


def test_take_first_handles_zero():
    assert take_first(["a", "b"], 0) == []
