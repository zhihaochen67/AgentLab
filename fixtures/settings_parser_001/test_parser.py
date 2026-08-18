from parser import parse_settings


def test_parse_settings_trims_keys_and_values():
    assert parse_settings("host = localhost, port = 8080") == {
        "host": "localhost",
        "port": "8080",
    }


def test_parse_settings_handles_single_pair():
    assert parse_settings("mode=debug") == {"mode": "debug"}


def test_parse_settings_handles_empty_text():
    assert parse_settings("") == {}
