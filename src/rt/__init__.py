from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("relational-transformer")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["RelationalTransformer", "__version__"]


def __getattr__(name):
    if name == "RelationalTransformer":
        from rt.model import RelationalTransformer

        return RelationalTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
