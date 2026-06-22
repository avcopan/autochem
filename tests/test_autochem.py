"""autochem tests."""

import autochem


def test_stub() -> None:
    """Stub test to ensure the test suite runs."""
    print(autochem.__version__)  # noqa: T201


def test__greet() -> None:
    """Test the greet function."""
    assert autochem.greet("World") == "Hello, World!"


def test__greet_jim() -> None:
    """Test the greet_jim function."""
    assert autochem.greet_jim() == "Hello, Jim!"
