"""Acceptance checks against the installed fork wheel, not a source-path shim."""

from importlib.metadata import distribution
from importlib.util import find_spec


def test_distribution_does_not_advertise_an_unimplemented_cli():
    package = distribution("openhcs-basicpy")
    assert not package.entry_points.select(group="console_scripts")
    assert find_spec("basicpy.__main__") is None
