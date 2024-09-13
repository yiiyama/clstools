"""Compute the observed limit on the signal strength parameter."""
from collections.abc import Callable, Sequence
from typing import Any, Optional
import logging
from multiprocessing import Pool
import time
import numpy as np
import h5py
from .test_statistic import calculate_pnllr

LOG = logging.getLogger('clstools')


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
    refine_tol: Optional[float] = None,
    parallelize: bool = False,
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
        refine_tol: If not None, refine the limit by binary search within the tolerance.
        parallelize: Whether to parallelize test statistic calculation over toys.
        output_path: Path of the file where all auxiliary data are stored.

    Returns:
        Observed limit on the signal strength at the given confidence level.
    """
    fitter_kwargs = fitter_kwargs or {}
    generator_kwargs = generator_kwargs or {}
    popt_null, _ = fitter(xdata, ydata, 0., *fitter_args, **fitter_kwargs)
    toys_null = generator(xdata, popt_null + (0.,), num_toys, *generator_args, **generator_kwargs)

    test_obs_arr = []
    popt_mu_arr = []
    test_nulls_arr = []
    toys_mu_arr = []
    test_mus_arr = []

    best_fit_obs = (None, None)
    best_fit_toys = [(None, None)] * num_toys

    def calculate_cls(mu_value):
        nonlocal best_fit_obs

        test_obs, popt_mu, best_fit = calculate_pnllr(fitter, xdata, ydata, mu_value,
                                                      args=fitter_args, kwargs=fitter_kwargs,
                                                      nll_glob=best_fit_obs[1])
        if best_fit_obs[1] is None and best_fit[0] < mu_value * 0.999:
            best_fit_obs = best_fit

        test_nulls = np.empty(num_toys)
        if parallelize:
            args = [(fitter, xdata, toy, mu_value, fitter_args, fitter_kwargs, best_fit[1])
                    for toy, best_fit in zip(toys_null, best_fit_toys)]
            with Pool() as pool:
                results = pool.starmap(calculate_pnllr, args)

            for itoy, (test, _, best_fit) in enumerate(results):
                test_nulls[itoy] = test
                if best_fit_toys[itoy][1] is None and best_fit[0] < mu_value * 0.999:
                    best_fit_toys[itoy] = best_fit
        else:
            for itoy, toy in enumerate(toys_null):
                test, _, best_fit = calculate_pnllr(fitter, xdata, toy, mu_value,
                                                    args=fitter_args, kwargs=fitter_kwargs,
                                                    nll_glob=best_fit_toys[itoy][1])
                test_nulls[itoy] = test
                if best_fit_toys[itoy][1] is None and best_fit[0] < mu_value * 0.999:
                    best_fit_toys[itoy] = best_fit

        toys_mu = generator(xdata, popt_mu + (mu_value,), num_toys, *generator_args,
                            **generator_kwargs)

        test_mus = np.empty(num_toys)
        if parallelize:
            args = [(fitter, xdata, toy, mu_value, fitter_args, fitter_kwargs) for toy in toys_mu]
            with Pool() as pool:
                results = pool.starmap(calculate_pnllr, args)

            for itoy, (test, _, _) in enumerate(results):
                test_mus[itoy] = test
        else:
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
        return p_s / (1. - p_b)

    start_time = time.time()
    mu_values = np.array(list(sorted(mu_values)))

    imu = 0
    while imu < len(mu_values):
        mu_value = mu_values[imu]
        cls = calculate_cls(mu_value)
        LOG.info('mu=%f, CLs=%f, %f s elapsed', mu_value, cls, time.time() - start_time)
        if cls < 1. - confidence_level:
            break
        imu += 1

    if refine_tol is not None:
        LOG.info('Refining CLs calculation..')

        if imu == 0:
            mu_min = 0.
            mu_max = mu_value
        elif imu == len(mu_values):
            additional_mu_values = np.array([mu_values[-1] * (2 ** ip) for ip in range(6)])
            imu = 1
            while imu < len(additional_mu_values):
                mu_value = additional_mu_values[imu]
                cls = calculate_cls(mu_value)
                LOG.info('mu=%f, CLs=%f, %f s elapsed', mu_value, cls, time.time() - start_time)
                if cls < 1. - confidence_level:
                    break
                imu += 1
            else:
                raise RuntimeError('Maximum mu value reached')

            mu_min, mu_max = additional_mu_values[imu - 1:imu + 1]
        else:
            mu_min, mu_max = mu_values[imu - 1:imu + 1]

        while mu_max - mu_min > refine_tol:
            mu_value = (mu_min + mu_max) * 0.5
            if mu_value < best_fit_obs[0]:
                best_fit_obs = (None, None)
            for itoy in range(num_toys):
                if mu_value < best_fit_toys[itoy][0]:
                    best_fit_toys[itoy] = (None, None)

            cls = calculate_cls(mu_value)
            LOG.info('mu=%f, CLs=%f, %f s elapsed', mu_value, cls, time.time() - start_time)
            if cls < 1. - confidence_level:
                mu_max = mu_value
            else:
                mu_min = mu_value

    if output_path:
        with h5py.File(output_path, 'w') as out:
            out.create_dataset('xdata', data=xdata)
            out.create_dataset('ydata', data=ydata)
            out.create_dataset('mu_values', data=mu_values)
            out.create_dataset('num_toys', data=num_toys)
            out.create_dataset('confidence_level', data=confidence_level)
            out.create_dataset('limit', data=mu_value)
            out.create_dataset('popt_null', data=popt_null)
            out.create_dataset('toys_null', data=toys_null)
            out.create_dataset('test_obs', data=test_obs_arr)
            out.create_dataset('popt_mu', data=popt_mu_arr)
            out.create_dataset('test_nulls', data=test_nulls_arr)
            out.create_dataset('toys_mu', data=toys_mu_arr)
            out.create_dataset('test_mus', data=test_mus_arr)

    return mu_value, cls
