from search import binary_search


def test_binary_search_finds_first_item():
    assert binary_search([2, 4, 6, 8, 10], 2) == 0


def test_binary_search_finds_middle_item():
    assert binary_search([2, 4, 6, 8, 10], 6) == 2


def test_binary_search_finds_last_item():
    assert binary_search([2, 4, 6, 8, 10], 10) == 4


def test_binary_search_returns_minus_one_when_missing():
    assert binary_search([2, 4, 6, 8, 10], 7) == -1
