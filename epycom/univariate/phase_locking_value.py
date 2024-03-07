# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
import numpy as np
import scipy.signal as sp

# Local imports
from ..utils.method import Method


def compute_plv_count(sig, fs=None, lowband=[4, 8], highband=[80, 150]):
    """
    Function to compute phase-locking value (PLV) of given signal

    Parameters
    ----------
    fs: float64
        frequency
    sig: numpy.ndarray
        signal from which PLV is computed
    lowband: list
            low frequency band boundaries [x, y], default [4, 8]
    highband: list
            high frequency band boundaries [x, y], default [80, 150]

    Returns
    -------
    PLV: numpy.complex128
        phase-locking value computed as np.mean(np.exp(1j*(phase1-phase2)))

    Example
    -------
    PLV = compute_plv_count(sig, fs)

    """

    order = 3
    nyq = fs * 0.5

    lowband = np.divide(lowband, nyq)
    highband = np.divide(highband, nyq)

    [b, a] = sp.butter(order, lowband, btype='bandpass', analog=False)
    low = sp.filtfilt(b, a, sig, axis=0)

    [b, a] = sp.butter(order, highband, btype='bandpass', analog=False)
    high = sp.filtfilt(b, a, sig, axis=0)

    # Extracting phase from the low frequency filtered analytic signal
    analytic_signal = sp.hilbert(low)
    phase1 = np.angle(analytic_signal)

    # Extracting amplitude from the high frequency filtered analytic signal
    amp_analytic_signal = sp.hilbert(high)
    amplitude = np.abs(amp_analytic_signal)

    second_analytic_signal = sp.hilbert(amplitude)
    phase2 = np.angle(second_analytic_signal)

    #  Counting phase locking value of a given signal
    plv = np.mean(np.exp(1j*(phase1-phase2)))

    return plv


class PhaseLockingValue(Method):

    algorithm = 'PHASE_LOCKING_VALUE'
    algorithm_type = 'univariate'
    version = '1.1.0'
    dtype = [('plv', 'complex64')]

    def __init__(self, **kwargs):
        """
        Phase locking value

        Parameters
        ----------
        fs: float64
            frequency
        sig: numpy.ndarray
            signal from which PLV is computed
        lowband: list
                low frequency band boundaries [x, y]
        highband: list
                high frequency band boundaries [x, y]
        """

        super().__init__(compute_plv_count, **kwargs)
