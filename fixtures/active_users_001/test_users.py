from users import active_usernames


def test_only_active_users_are_returned():
    users = [
        {"name": "Ada", "active": True},
        {"name": "Grace", "active": False},
        {"name": "Linus", "active": True},
    ]

    assert active_usernames(users) == ["Ada", "Linus"]


def test_missing_active_flag_is_not_active():
    assert active_usernames([{"name": "Sam"}]) == []


def test_empty_users_returns_empty_list():
    assert active_usernames([]) == []
