# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
import numpy as np

# Local imports
from .hjorth_mobility import compute_hjorth_mobility
from ..utils.method import Method


def compute_hjorth_complexity(signal, fs=None):
    """
    Compute Hjorth complexity of time series

    Parameters
    ----------
    signal: np.array
        Signal to analyze, time series (array, int, float)
    fs: float
        Sampling frequency of the time series

    Returns
    -------
    hjorth_complexity: float

    Example
    -------
    hjorth_complexity = compute_hjorth_complexity(signal)

    Note
    ----
    fs (sampling frequency) is left for backwards compatibility, but as was
    shown, the ressult of Hjort complexity is not dependent on sampling
    frequency of given signal
    """
    variancex = np.var(signal)
    if variancex == 0:
        return float('NaN')
    variancedx = np.var(np.diff(signal))
    if variancedx == 0:
        return float('NaN')
    # if the variance of original signal is zero, the varianceddx would be also
    # zero and division 0/0 is undefined
    varianceddx = np.var(np.diff(signal, n=2))

    hjorth_complexity = np.sqrt(variancex*varianceddx)/variancedx
    return hjorth_complexity


class HjorthComplexity(Method):

    algorithm = 'HJORTH_COMPLEXITY'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('hjorth_complexity', 'float32')]

    def __init__(self, **kwargs):
        """
        Hjorth complexity of time series

        Parameters
        ----------
        fs: float
            Sampling frequency of the time series

        Note
        ----
        fs (sampling frequency) is left for compatibility, but as was showed,
        the ressult of Hjort complexity is not dependent on sampling frequency
        of given signal
        """

        super().__init__(compute_hjorth_complexity, **kwargs)
