"""Main module."""

import sys


def main(argv: list[str] | None = None):
    """Command-line interface's entrypoint.

    Args:
        argv (list, optional): Argument values. Defaults to None.
    """
    if argv is None:
        argv = sys.argv[1:]
    print("Hello World!")
    print(argv)
