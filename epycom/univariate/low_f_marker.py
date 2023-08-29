# Third party imports
import numpy as np
# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
import scipy.signal as sp

# Local imports
from ..utils.method import Method


def compute_low_f_marker(sig, fs=None):
    """
    Function to compute power ratio of two signal windows filtered on different
    Frequencies based on Lundstrom et al. 2021
    https://www.medrxiv.org/content/10.1101/2021.06.04.21258382v1.full.pdf

    Parameters
    ----------
    fs: int
            Sampling frequency

    Returns
    --------
    low_f_marker: float32
        returns median of given time window 
    """
    order = 1

    lowband=[0.02, 0.5]
    highband=[2.0, 4.0]

    nyq = fs * 0.5

    lowband = np.divide(lowband, nyq)
    highband = np.divide(highband, nyq)

    [b, a] = sp.butter(order, lowband, btype='bandpass', analog=False)
    infra_signal = sp.filtfilt(b, a, sig, axis=0)

    [b, a] = sp.butter(order, highband, btype='bandpass', analog=False)
    main_signal = sp.filtfilt(b, a, sig, axis=0)

    low_f_power_ratio = infra_signal**2/main_signal**2
    low_f_marker = np.median(low_f_power_ratio)
    
    return low_f_marker

class LowFreqMarker(Method):

    algorithm = 'LOW_FREQUENCY_MARKER'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('lowFreqMark', 'float32')]

    def __init__(self, **kwargs):
        """
        Modulation Index

        Parameters
        ----------
        fs: int
            Sampling frequency
        """

        super().__init__(compute_low_f_marker, **kwargs)