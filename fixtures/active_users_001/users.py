def active_usernames(users):
    """Return usernames for active users in input order."""
    return [user["name"] for user in users if not user.get("active", False)]
