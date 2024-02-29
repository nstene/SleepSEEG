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


def compute_lincorr(sig, lag=0, lag_step=0):
    """
    Linear correlation (Pearson's coefficient) between two time series

    If lag and lag_step is not 0, calculates evolution of correlation by
    shifting the sig[1] from from negative lag to positive lag.
    From list of correlations takes the max correlation (best fit).

    Parameters:
    -----------
    sig: np.array
        2D numpy array of shape (signals, samples), time series (int, float)
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift

    Returns
    -------
    max_lincorr: float
        maximum linear correlation in shift
        max_lincorr = 1:    perfect conformity sig[1] and sig[0]
        max_lincorr = -1:   opposite signals
    k: int
        shift of maximum coherence in samples,
        value in range <-lag,+lag> (int)
        k > 0: sig[1] -> sig[0]
        k = 0: no shift in signals
        k < 0: sig[0] -> sig[1]

    Example
    -------
    max_lincorr,k = compute_lincorr(sig, 200, 20)
    """

    if type(sig) != np.ndarray:
        raise TypeError(f"Signals have to be in numpy arrays!")

    if sig.ndim != 2:
        raise TypeError(f"The array must have two dimensions not {sig.ndim}!")

    if lag_step == 0:
        lag_step = 1
    nstep_lag = int(lag * 2 / lag_step)

    sig1_w = sig[0]
    sig2_w = sig[1]

    sig1_wl = sig1_w[lag:len(sig1_w) - lag]

    lincorr = []
    for i in range(0, nstep_lag + 1):
        ind1 = i * lag_step
        ind2 = ind1 + len(sig1_wl)

        sig2_wl = sig2_w[ind1:ind2]

        corr_val = np.corrcoef(sig1_wl, sig2_wl)
        lincorr.append(corr_val[0][1])

    return max(lincorr, key=abs), lag-(lincorr.index(max(lincorr, key=abs)))*lag_step 


class LinearCorrelation(Method):
    """
    Linear correlation (Pearson's coefficient) between two time series

    If lag and lag_step is not 0, calculates evolution of correlation by
    shifting the sig[1] from from negative lag to positive lag.
    From list of correlations takes the max correlation (best fit).

    Parameters
    ----------
    sig: np.array
        2D numpy array of shape (signals, samples),
        time series (int, float)
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift

    """

    algorithm = 'LINEAR_CORRELATION'
    algorithm_type = 'bivariate'
    is_directional = True
    version = '2.0.0'
    dtype = [('max_corr', 'float32'),
             ('k', 'int')]

    def __init__(self, **kwargs):
        """
        Linear correlation (Pearson's coefficient) between two time series

        If lag and lag_step is not 0, calculates evolution of correlation by
        shifting the sig[1] from from negative lag to positive lag.
        From list of correlations takes the max correlation (best fit).

        Parameters
        ----------
        sig: np.array
            2D numpy array of shape (signals, samples), 
            time series (int, float)
        lag: int
            negative and positive shift of time series in samples
        lag_step: int
            step of shift

        """
        super().__init__(compute_lincorr, **kwargs)
