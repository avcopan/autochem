"""Layer 2 functions.

Layering is enforced by import-linter.
"""

from .layer1 import greet


def greet_jim() -> str:
    """Greet Jim specifically.

    Returns
    -------
        A greeting message for Jim.
    """
    return greet("Jim")
