"""Configuration parsing helpers shared by draft / verify runtimes."""


def _to_int(value, default):
    """Convert a value to int and fall back to the provided default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_optional_int(value, default=None):
    """Convert a value to int, but allow missing values to remain unset."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value, default=True):
    """Convert common string / bool forms into a Python boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return default


def _to_int_list(value, default):
    """
    Convert a config value into a sorted list of positive ints.

    Accepts:
    - Python lists / tuples / sets
    - comma-separated strings such as "8,12,16,24"
    """
    if value is None:
        items = list(default)
    elif isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    parsed = []
    for item in items:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0:
            parsed.append(number)
    parsed = sorted(set(parsed))
    return parsed or list(default)


to_int = _to_int
to_optional_int = _to_optional_int
to_bool = _to_bool
to_int_list = _to_int_list

as_bool = _to_bool
as_int = _to_int
as_optional_int = _to_optional_int
as_int_list = _to_int_list
