"""Parameter definitions for model construction.

A Parameter represents a single floating-point degree of freedom in the
statistical model. It carries a name, nominal value, bounds, and a
flag indicating whether the minimiser should float or fix it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Parameter:
    """A single model parameter.

    Parameters
    ----------
    name : str
        Unique identifier used as the keyword argument in minimisation.
    value : float
        Starting (nominal) value.
    bounds : tuple[float, float] or None
        (lower, upper) allowed range. ``None`` means unbounded.
    fixed : bool
        If ``True``, the minimiser will not vary this parameter.
    """

    name: str
    value: float = 1.0
    bounds: tuple[float, float] | None = None
    fixed: bool = False

    # --- convenience -----------------------------------------------------------

    def __post_init__(self) -> None:
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo >= hi:
                raise ValueError(
                    f"Parameter '{self.name}': lower bound {lo} must be "
                    f"strictly less than upper bound {hi}"
                )
            if not (lo <= self.value <= hi):
                raise ValueError(
                    f"Parameter '{self.name}': value {self.value} is outside "
                    f"bounds ({lo}, {hi})"
                )

    def clone(self, **overrides) -> Parameter:
        """Return a copy, optionally overriding fields."""
        kw = {
            "name": self.name,
            "value": self.value,
            "bounds": self.bounds,
            "fixed": self.fixed,
        }
        kw.update(overrides)
        return Parameter(**kw)

    # --- iminuit helpers -------------------------------------------------------

    def to_minuit_init(self) -> dict:
        """Return a dict suitable for passing to ``iminuit.Minuit`` as kwargs."""
        d: dict = {self.name: self.value}
        if self.bounds is not None:
            d[f"limit_{self.name}"] = self.bounds
        if self.fixed:
            d[f"fix_{self.name}"] = True
        return d
