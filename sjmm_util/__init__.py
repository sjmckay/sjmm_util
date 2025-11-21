from . import config, catalog, general_utils, datatools, line_utils, scuba2_utils

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sjmm_util")
except PackageNotFoundError:
    __version__ = "0.0.0"