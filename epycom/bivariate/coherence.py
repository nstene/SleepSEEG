# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np

# Local imports
from ..utils.method import Method

def compute_coherence(sig, fs=5000):
    """
    Magnitude squared coherence between two time series

    Parameters
    ----------
    sig: np.array
        2D numpy array of shape (signals, samples), time series (int, float)
    fs: int, float
        sampling frequency in Hz

    Returns
    -------
    coherence: float

    Example
    -------
    c = compute_coherence(sig)
    """
    
    if type(sig) != np.ndarray:
        raise TypeError("Signals have to be in numpy arrays!")
    
    sig0 = sig[0]
    sig1 = sig[1]
    
    if len(sig0)/2 % 2 != 0:
        sig0 = np.append(sig0,sig0[-1])
        sig1 = np.append(sig1,sig1[-1])
    
    ps0 = np.abs(np.fft.fft(sig0))
    ps1 = np.abs(np.fft.fft(sig1))
    
    corr_val = np.corrcoef(ps0, ps1)
    
    return corr_val[0][1]

class Coherence(Method):

    algorithm = 'COHERENCE'
    algorithm_type = 'bivariate'
    version = '1.0.0'
    dtype = [('coh', 'float32')]


    def __init__(self, **kwargs):
        """
        Magnitude squared coherence between two time series

         Parameters
         ----------
         sig: np.array
             2D numpy array of shape (signals, samples), time series (int, float)
         fs: int, float
             sampling frequency in Hz
        """

        super().__init__(compute_coherence, **kwargs)
