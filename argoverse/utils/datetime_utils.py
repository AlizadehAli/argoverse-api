# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Datetime utility functions."""

import datetime


def generate_datetime_string() -> str:
    """Generate a formatted datetime string.

    Returns:
        String with of the format YYYY_MM_DD_HH_MM_SS with 24-hour time used
    """
    return ("%4i_%2i_%2i_%2i_%2i_%2i" % datetime.datetime.now())
