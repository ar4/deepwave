"""Measure the runtime of propagators"""
from timeit import repeat
import torch
import numpy as np
from deepwave.scalar import Propagator
import test_scalar


def _versions():
    """Returns default versions."""
    return [{'name': 'Scalar', 'propagator': Propagator,
             'prop_kwargs': None}]


def scalarprop_wrapper(propagator):
    """Wrapper around a wrapper around the propagator to test.

    This double wrapper is necessary so that we can specify which propagator
    to use.
    """
    def scalarprop(model, dx, dt, sources, receiver_locations, grad=False,
                   forward_true=None, prop_kwargs=None):
        """Wrapper around the propagator that measures the runtime."""

        if propagator is None:
            raise ValueError("Must specify propagator")

        if prop_kwargs is None:
            prop_kwargs = {}

        def closure():
            """Closure over variables so they can be used in repeat below."""
            prop = propagator(model, dx, **prop_kwargs)
            receiver_amplitudes = prop.forward(sources['amplitude'],
                                               sources['locations'],
                                               receiver_locations, dt)

            if grad:
                l = torch.nn.MSELoss()(receiver_amplitudes,
                                       forward_true)
                l.backward()

        return np.min(repeat(closure, number=1))

    return scalarprop


def time_version(propagator, model, prop_kwargs=None, model_kwargs=None):
    """Measure the runtime of the specified version using the specified model.

    Args:
        propagator: The propagator object to use
        model: The model (such as one from test_scalar) to use

    Returns:
        Runtime in seconds
    """
    if model_kwargs is None:
        model_kwargs = {}
    scalarprop = scalarprop_wrapper(propagator)
    _, runtime = model(propagator=scalarprop, prop_kwargs=prop_kwargs,
                       **model_kwargs)

    return runtime


def run_timing(versions=None, model=None, model_kwargs=None):
    """Loop over specified versions and measure runtime.

    Args:
        versions: List of dictionaries that contain the keys
            name: The name to use when reporting runtime
            propagator: The propagator object to use
            prop_kwargs: Keyword arguments to the propagator
        model: The model (such as one from test_scalar) to use
        model_kwargs: Keyword arguments to the model

    Returns:
        List of dictionaries with the keys
            name: The name provided for the propagator
            time: The measured runtime in seconds
    """

    if versions is None:
        versions = _versions()

    if model is None:
        model = test_scalar.run_direct_2d

    timing = []
    for version in versions:
        runtime = time_version(version['propagator'], model,
                               prop_kwargs=version['prop_kwargs'],
                               model_kwargs=model_kwargs)
        timing.append({'name': version['name'], 'time': runtime})
    print(timing)
    return timing


if __name__ == '__main__':

    run_timing()
