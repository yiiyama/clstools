"""Compute the observed limit on the signal strength parameter."""
from collections.abc import Callable, Sequence
from typing import Any, Optional
import numpy as np
import h5py
from .test_statistic import calculate_pnllr


def get_observed_limit(
    mu_values: Sequence[float],
    fitter: Callable,
    generator: Callable,
    xdata: Any,
    ydata: Any,
    fitter_args: tuple[Any, ...] = (),
    fitter_kwargs: Optional[dict[str, Any]] = None,
    generator_args: tuple[Any, ...] = (),
    generator_kwargs: Optional[dict[str, Any]] = None,
    num_toys: int = 200,
    confidence_level: float = 0.95,
    output_path: Optional[str] = None
) -> float:
    """Compute the observed limit.

    Args:
        mu_values: Signal strength values to try.
        minimize_nll: Function to minimize the negative log-likelihood given the input xdata, ydata,
            signal strength (value or bounds), and arguments.

                ``minimize_nll(xdata, ydata, mu, *args, **kwargs) -> popt, nll``

            where ``mu`` is either a float scalar or (min, max). ``popt`` is the tuple of best-fit
            model parameter values, with the signal strength as the last entry when the bounds tuple
            is passed as ``mu``. ``nll`` is the best-fit NLL value.
        generate_toys: Function to generate toy datasets given the input xdata, model parameters,
            and arguments.

                ``generate_toys(xdata, params, num_toys, *args, **kwargs) -> list of ydata``

            where ``params`` is a tuple of model parameters with the signal strength as the last
            entry.

        xdata: Independent variable of the model.
        ydata: Observed data.
        args: Additional positional arguments to be passed to the model functions.
        kwargs: Additional keyword arguments to be passed to the model functions.
        num_toys: Size of toy dataset to generate for each signal strength value.
        confidence_level: Confidence level.
        output_path: Path of the file where all auxiliary data are stored.

    Returns:
        Observed limit on the signal strength at the given confidence level.
    """
    popt_null, _ = fitter(xdata, ydata, 0., *fitter_args, **fitter_kwargs)
    toys_null = generator(xdata, popt_null + (0.,), num_toys, *generator_args, **generator_kwargs)

    nll_glob_obs = None
    nll_glob_toys = [None] * num_toys

    limit = None

    test_obs_arr = []
    popt_mu_arr = []
    test_nulls_arr = []
    toys_mu_arr = []
    test_mus_arr = []

    for mu_value in sorted(mu_values):
        test_obs, popt_mu, best_fit = calculate_pnllr(fitter, xdata, ydata, mu_value,
                                                      args=fitter_args, kwargs=fitter_kwargs,
                                                      nll_glob=nll_glob_obs)
        if nll_glob_obs is None and best_fit[0] < mu_value * 0.999:
            nll_glob_obs = best_fit[1]

        test_nulls = np.empty(num_toys)
        for itoy, toy in enumerate(toys_null):
            test, _, best_fit = calculate_pnllr(fitter, xdata, toy, mu_value,
                                                args=fitter_args, kwargs=fitter_kwargs,
                                                nll_glob=nll_glob_toys[itoy])
            test_nulls[itoy] = test
            if nll_glob_toys[itoy] is None and best_fit[0] < mu_value * 0.999:
                nll_glob_toys[itoy] = best_fit[1]

        toys_mu = generator(xdata, popt_mu + (mu_value,), num_toys, *generator_args,
                            **generator_kwargs)

        test_mus = np.empty(num_toys)
        for itoy, toy in enumerate(toys_mu):
            test_mus[itoy] = calculate_pnllr(fitter, xdata, toy, mu_value,
                                             args=fitter_args, kwargs=fitter_kwargs)[0]

        if output_path:
            test_obs_arr.append(test_obs)
            popt_mu_arr.append(popt_mu)
            test_nulls_arr.append(test_nulls)
            toys_mu_arr.append(toys_mu)
            test_mus_arr.append(test_mus)

        p_s = np.nonzero(test_mus >= test_obs)[0].shape[0] / num_toys
        p_b = 1. - np.nonzero(test_nulls >= test_obs)[0].shape[0] / num_toys
        cls = p_s / (1. - p_b)

        if cls < 1. - confidence_level:
            limit = mu_value
            break

    if output_path:
        with h5py.File(output_path, 'w') as out:
            out.create_dataset('xdata', data=xdata)
            out.create_dataset('ydata', data=ydata)
            out.create_dataset('mu_values', data=np.array(list(sorted(mu_values))))
            out.create_dataset('num_toys', data=num_toys)
            out.create_dataset('confidence_level', data=confidence_level)
            out.create_dataset('popt_null', data=popt_null)
            out.create_dataset('toys_null', data=toys_null)
            out.create_dataset('test_obs', data=test_obs_arr)
            out.create_dataset('popt_mu', data=popt_mu_arr)
            out.create_dataset('test_nulls', data=test_nulls_arr)
            out.create_dataset('toys_mu', data=toys_mu_arr)
            out.create_dataset('test_mus', data=test_mus_arr)

    return limit, cls
