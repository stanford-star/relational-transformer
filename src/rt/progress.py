import re

BOLD = "\033[1m"
RESET = "\033[0m"

_NUMERIC = re.compile(r"^[+-]?\d")


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def fmt_bytes(n):
    return f"{n / (1024**3):.2f}GiB"


_widths: dict[str, dict[str, int]] = {}


def log(*, indent=0, **fields):
    widths = _widths.setdefault(",".join(fields), {})
    parts = []
    for k, v in fields.items():
        v = str(v)
        widths[k] = max(widths.get(k, 0), len(v))
        v = v.ljust(widths[k])
        parts.append(
            f"{k}: {BOLD}{v}{RESET}" if _NUMERIC.match(v.strip()) else f"{k}: {v}"
        )
    print(("  " * indent + "  ".join(parts)).rstrip(), flush=True)
