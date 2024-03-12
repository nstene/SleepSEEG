# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

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
    Function to compute median of power ratio of two signals obtained form 
    input signal by Butterworth filter on lowband=[0.02, 0.5] and 
    highband=[2.0, 4.0] frequencies.
    
    Based on Lundstrom et al. 2021
    https://www.medrxiv.org/content/10.1101/2021.06.04.21258382v1.full.pdf

    Parameters
    ----------
    signal: np.array
            Signal to analyze, time series (array, int, float)
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

    sos_low = sp.butter(order, lowband, btype='bandpass', output='sos', 
                        analog=False)
    infra_signal = sp.sosfiltfilt(sos_low, sig, axis=0)

    sos_high = sp.butter(order, highband, btype='bandpass', output='sos', 
                         analog=False)
    main_signal = sp.sosfiltfilt(sos_high, sig, axis=0)

    low_f_power_ratio = infra_signal**2/main_signal**2
    low_f_marker = np.median(low_f_power_ratio)
    
    return low_f_marker

class LowFreqMarker(Method):

    algorithm = 'LOW_FREQUENCY_MARKER'
    algorithm_type = 'univariate'
    version = '1.0.1'
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
