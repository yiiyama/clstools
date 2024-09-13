"""Functions to calculate various test statistics."""
from collections.abc import Callable
from typing import Any, Optional


def calculate_pnllr(
    minimize_nll: Callable,
    xdata: Any,
    ydata: Any,
    mu: float,
    args: tuple[Any, ...] = (),
    kwargs: Optional[dict[str, Any]] = None,
    nll_glob: Optional[float] = None
) -> tuple[float, tuple[float, ...], float]:
    """Calculate the profiled negative log likelihood ratio.

    Args:
        minimize_nll: Function to minimize the negative log-likelihood given the input xdata, ydata,
            signal strength (value or bounds), and arguments.

                ``minimize_nll(xdata, ydata, mu, *args, **kwargs) -> popt, nll``

            where ``mu`` is either a float scalar or (min, max). ``popt`` is the tuple of best-fit
            model parameter values, with the signal strength as the last entry when the bounds tuple
            is passed as ``mu``. ``nll`` is the best-fit NLL value.
        xdata: Independent variable of the model.
        ydata: Observed data.
        args: Additional positional arguments to be passed to the model functions.
        kwargs: Additional keyword arguments to be passed to the model functions.
        nll_glob: Global best-fit NLL value. If given, assumed to correspond to a muhat less than
            mu, and the mu-profiling fit is skipped.

    Returns:
        - Test statistic value
        - The best-fit parameters for the given mu
        - Best-fit signal strength (muhat) and NLL value with constraint 0 <= muhat <= mu
    """
    popt_cond, nll_cond = minimize_nll(xdata, ydata, mu, *args, **kwargs)
    if nll_glob is None:
        popt_glob, nll_glob = minimize_nll(xdata, ydata, (0., mu), *args, **kwargs)
        best_fit = (popt_glob[-1], nll_glob)
    else:
        best_fit = None

    return 2. * (nll_cond - nll_glob), popt_cond, best_fit
