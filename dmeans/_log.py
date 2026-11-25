"""Shared logger. Every module logs under the "dmeans" namespace, in English."""

import logging
import os

LOG = logging.getLogger("dmeans")


def get(name: str) -> logging.Logger:
    return LOG.getChild(name)


def setup(level: str | None = None) -> None:
    """Send dmeans logs to stderr. Level comes from DMEANS_LOG (default INFO)."""
    level = (level or os.environ.get("DMEANS_LOG", "INFO")).upper()
    if not LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        LOG.addHandler(h)
    LOG.setLevel(level)
    LOG.propagate = False
