"""Human-readable number formatting shared by agents."""

import math


def usd(value: float) -> str:
    if abs(value) >= 1 or value == 0:
        return f"${value:,.2f}"
    decimals = 3 - math.floor(math.log10(abs(value)))  # keep ~4 significant digits
    return f"${value:.{decimals}f}".rstrip("0")


def usd_zone(lower: float, upper: float) -> str:
    """A price zone as "~$lower–$upper", or "~$price" when both bounds are the same price."""
    if usd(lower) == usd(upper):
        return f"~{usd(lower)}"
    return f"~{usd(lower)}–{usd(upper)}"


def usd_compact(value: float) -> str:
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return usd(value)
