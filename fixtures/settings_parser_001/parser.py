def parse_settings(text):
    """Parse comma-separated key=value settings with surrounding whitespace."""
    settings = {}
    for item in text.split(","):
        key, value = item.split("=")
        settings[key] = value
    return settings
