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
from ..utils.method import Method


def compute_hjorth_mobility(signal, fs=5000):
    """
    Function to compute Hjorth mobility of time series

    Parameters
    ----------
    signal: np.array
        Signal to analyze, time series (array, int, float)
    fs: float
        Sampling frequency of the time series

    Returns
    -------
    hjorth_mobility: float

    Example
    -------
    hjorth_mobility = compute_hjorth_mobility(signal, 5000)

    Note
    ----
    Result is frequency dependent
    """

    variancex = signal.var(ddof=1)
    # diff signal is one sample shorter
    if variancex == 0:
        return float('NaN')
    # if the variance of original signal is zero, the variancedx would be also 
    # zero and division 0/0 is undefined
    variancedx = np.var(np.diff(signal) * fs, ddof=1)
    # compute variance with degree of freedom=1 => The mean is normally
    # calculated as x.sum() / N, where N = len(x). If, however, ddof is
    # specified, the divisor N - ddof is used instead.

    hjorth_mobility = np.sqrt(variancedx / variancex)
    return hjorth_mobility


class HjorthMobility(Method):

    algorithm = 'HJORTH_MOBILITY'
    algorithm_type = 'univariate'
    version = '1.1.0'
    dtype = [('hjorth_mobility', 'float32')]

    def __init__(self, **kwargs):
        """
        Hjorth mobility of time series

        Parameters
        ----------
        fs: float
            Sampling frequency of the time series
        """

        super().__init__(compute_hjorth_mobility, **kwargs)
