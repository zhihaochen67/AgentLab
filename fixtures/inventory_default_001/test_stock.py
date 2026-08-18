from stock import quantity_for


def test_existing_sku_returns_stored_quantity():
    assert quantity_for({"A-1": 7}, "A-1") == 7


def test_missing_sku_returns_zero():
    assert quantity_for({"A-1": 7}, "B-2") == 0


def test_explicit_zero_is_preserved():
    assert quantity_for({"A-1": 0}, "A-1") == 0
