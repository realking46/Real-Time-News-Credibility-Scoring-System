"""
Smoke tests — these run in CI from day 1.
Real tests are added as each pipeline is built.
"""


def test_placeholder():
    """Placeholder so pytest always finds at least one test."""
    assert 1 + 1 == 2
