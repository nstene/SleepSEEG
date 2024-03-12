# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
import numpy as np
from scipy.signal import coherence

# Local imports
from ..utils.method import Method


def compute_coherence(sig, fs=5000, fband=[1.0, 4.0], lag=0, lag_step=0,
                      fft_win=5000):
    """
    Magnitude squared coherence between two time series (raw,
    not filtered signals)

    If fft_win (fast Fourier transform window) is not 0, calculates evolution 
    of coherence

    If fft_win>len(sig) or fft_win<=0, calculates only one coherence value

    When lag and lag_step is not 0, shifts the sig[1] from negative
    to positive lag and takes the max coherence (best fit)

    Parameters
    ----------
    sig: np.array
        2D numpy array of shape (signals, samples), time series (int, float)
    fs: int, float
        sampling frequency in Hz
        for good ressults, fs > 250 is recommended
    fband: list
        frequency range in Hz (float)
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift in samples
    fft_win: int
        length of fft window in samples

    Returns
    -------
    max_coh: float
        maximum coherence in shift
    k: int
        shift of maximum coherence in samples,
        value in range <-lag,+lag> (int)
        k > 0: sig[1] -> sig[0]
        k = 0: no shift in signals
        k < 0: sig[0] -> sig[1]

    Example
    -------
    max_coh,k = compute_coherence(sig, fs=5000, fband=[1.0,4.0], lag=0,
                                                lag_step=0, fft_win=5000)
    """

    if type(sig) != np.ndarray:
        raise TypeError("Signals have to be in numpy arrays!")

    if sig.ndim != 2:
        raise TypeError(f"The array must have two dimensions not {sig.ndim}!")

    if lag_step == 0:
        lag_step = 1
    nstep_lag = int(lag * 2 / lag_step)

    # fft_win = int(fft_win*fs) # this line is not needed due to change fft_win 
    # #                             from sec to samples
    hz_bins = fft_win/fs
    fc1 = int(fband[0]*hz_bins)
    fc2 = int(fband[1]*hz_bins)

    sig1_w = sig[0]
    sig2_w = sig[1]

    sig1_wl = sig1_w[lag:len(sig1_w) - lag]

    coh_win = []
    for i in range(0, nstep_lag + 1):
        ind1 = i * lag_step
        ind2 = ind1 + len(sig1_wl)

        sig2_wl = sig2_w[ind1:ind2]

        f, coh = coherence(sig1_wl, sig2_wl, fs, nperseg=fft_win)
        coh_win.append(np.mean(coh[fc1:fc2]))

    return np.max(coh_win), lag-(coh_win.index(max(coh_win)))*lag_step 


class Coherence(Method):

    algorithm = 'COHERENCE'
    algorithm_type = 'bivariate'
    is_directional = True
    version = '2.0.0'
    dtype = [('max_coh', 'float32'),
             ('k', 'int')]


    def __init__(self, **kwargs):
        """
        Magnitude squared coherence between two time series (raw,
        not filtered signals)

        If fft_win (fast Fourier transform window) is not 0, calculates 
        evolution of coherence

        If win>len(sig) or win<=0, calculates only one coherence value

        If lag and lag_step is not 0, shifts the sig[1] from negative
        to positive lag and takes the max coherence (best fit)

        Parameters
        ----------
        sig: np.array
            2D numpy array of shape (signals, samples), 
            time series (int, float)
        fs: int, float
            sampling frequency in Hz
        fband: list
            frequency range in Hz (float)
        lag: int
            negative and positive shift of time series in samples
        lag_step: int
            step of shift in samples
        fft_win: int
            length of fft window in samples
        """

        super().__init__(compute_coherence, **kwargs)
