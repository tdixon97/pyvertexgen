from __future__ import annotations

import argparse
import logging

import colorlog


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="pyvertexgen",
        description="%(prog)s command line interface",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )

    


    # log handler

    args = parser.parse_args()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )
    logger = logging.getLogger("pyvertexgen")
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)