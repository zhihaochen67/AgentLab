from pricing import qualifies_for_discount


def test_total_at_threshold_qualifies():
    assert qualifies_for_discount(100) is True


def test_total_above_threshold_qualifies():
    assert qualifies_for_discount(125) is True


def test_total_below_threshold_does_not_qualify():
    assert qualifies_for_discount(99) is False
