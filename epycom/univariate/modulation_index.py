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


def compute_mi_count(sig, nbins=18):
    """
    Function to compute modulation index (MI) of given signal

    Parameters
    ----------
    sig: numpy.ndarray
        signal from which MI is computed
    nbins: int
        number of bins in which signal will be separated, can affecct the 
        result, default is 18

    Returns
    -------
    MI: float64
        modulation index computed as KL/np.log(nbins)

    Example
    -------
    MI = compute_mi_count(sig)

    """

    size = 2 * np.pi / nbins
    position = np.zeros(nbins)
    mean_amp = np.zeros(nbins)

    # Binning the phases
    for bins in range(0, nbins):
        position[bins] = -np.pi + bins * size

    f_sig = sp.hilbert(sig)
    ampl = np.abs(f_sig)
    ph = np.angle(f_sig)

    # Computing average amplitude in each bin
    for j in range(0, nbins):
        ampls = ampl[np.where((position[j] <= ph)&(ph < position[j] + size))]
        mean_amp[j] = np.mean(ampls)

    # Normalizing amplitudes
    p = mean_amp / np.sum(mean_amp)

    # Computing Shannon entropy
    H = -np.sum(p * np.log(p))

    # Computing Kullback–Leibler distance
    KL = np.log(nbins) - H

    # Final calculation of MI
    return KL / np.log(nbins)


class ModulationIndex(Method):

    algorithm = 'MODULATION_INDEX'
    algorithm_type = 'univariate'
    version = '1.0.1'
    dtype = [('mi', 'float32')]

    def __init__(self, **kwargs):
        """
        Modulation Index

        Parameters
        ----------
        sig: numpy.ndarray
            signal from which MI is computed
        nbins: int
            number of bins in which signal will be separated, can affecct the 
            result, default is 18
        """

        super().__init__(compute_mi_count, **kwargs)
