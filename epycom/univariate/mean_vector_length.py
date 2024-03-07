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


def compute_mvl_count(sig, fs, lowband=[4, 8], highband=[80, 150]):
    """
    Function to compute mean vector lenght (MVL) of given signal

    Parameters
    ----------
    fs: float64
        frequency
    sig: numpy.ndarray
        signal from which MVL is computed
    lowband: list
            low frequency band boundaries [x, y], default [4, 8]
    highband: list
            high frequency band boundaries [x, y], default [80, 150]

    Returns
    -------
    mvl: numpy.complex128
        MVL of given signal

    Example
    -------
    MVL = compute_mvl_count(sig, 5000.0)

    References
    ----------

    R. T. Canolty et al. ,High Gamma Power Is Phase-Locked to Theta 
    Oscillations in Human Neocortex.Science313,1626-1628(2006).
    DOI:10.1126/science.1128115
    https://www.science.org/doi/10.1126/science.1128115?ijkey=3426de4d785c48a29139c7352ea398ff3947c1fe&keytype2=tf_ipsecsha

    Quantification of Phase-Amplitude Coupling in Neuronal Oscillations: 
    Comparison of Phase-Locking Value, Mean Vector Length, and Modulation Index
    Mareike J. Hülsemann, Dr. rer. nat, Ewald Naumann, Dr. rer. nat, Björn 
    Rasch
    bioRxiv 290361; doi: https://doi.org/10.1101/290361

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
    phase = np.angle(analytic_signal)

    # Extracting amplitude from the high frequency filtered analytic signal
    amp_analytic_signal = sp.hilbert(high)
    amplitude = np.abs(amp_analytic_signal)

    # Counting mean vector length of a given signal
    mvl = amplitude * np.exp(1j*phase)

    return np.mean(mvl)


class MeanVectorLength(Method):

    algorithm = 'MEAN_VECTOR_LENGTH'
    algorithm_type = 'univariate'
    version = '1.1.0'
    dtype = [('mvl', 'complex64')]

    def __init__(self, **kwargs):
        """
        Mean vector length

        Parameters
        ----------
        fs: float64
            frequency
        sig: numpy.ndarray
            signal from which MVL is computed
        lowband: list
                low frequency band boundaries [x, y]
        highband: list
                high frequency band boundaries [x, y]
        """

        super().__init__(compute_mvl_count, **kwargs)
