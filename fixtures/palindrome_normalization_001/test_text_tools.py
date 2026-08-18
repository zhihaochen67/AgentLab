from text_tools import is_palindrome


def test_palindrome_ignores_spaces_and_case():
    assert is_palindrome("Never odd or even") is True


def test_palindrome_ignores_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama!") is True


def test_non_palindrome_is_false():
    assert is_palindrome("AgentLab") is False
