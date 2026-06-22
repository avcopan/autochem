"""Layer 1 functions.

Layering is enforced by import-linter.
"""


def greet(name: str) -> str:
    """Greet a person by their name.

    Parameters
    ----------
    name :
        The name of the person to greet.

    Returns
    -------
        A greeting message.
    """
    return f"Hello, {name}!"
