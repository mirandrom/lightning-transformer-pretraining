import os

from .utils.logging import get_logger

logger = get_logger(__name__)

# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_LTP_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "ltp")
LTP_CACHE_HOME = os.path.expanduser(os.getenv("LTP_CACHE", DEFAULT_LTP_CACHE_HOME))

DEFAULT_LTP_DATA_CACHE = os.path.join(LTP_CACHE_HOME, "data")
LTP_DATA_CACHE = os.getenv("LTP_DATA_CACHE", DEFAULT_LTP_DATA_CACHE)

DEFAULT_LTP_RUN_CACHE = os.path.join(LTP_CACHE_HOME, "runs")
LTP_RUN_CACHE = os.getenv("LTP_RUN_CACHE", DEFAULT_LTP_RUN_CACHE)