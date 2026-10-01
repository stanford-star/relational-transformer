from importlib.metadata import PackageNotFoundError, version

from rt.model import RelationalTransformer

try:
    __version__ = version("relational-transformer")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["RelationalTransformer", "__version__"]
