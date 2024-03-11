# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
import warnings

# Third party imports
import numpy as np
from numba import njit

# Local imports
from ..utils.method import Method


@njit('f8(f8[:], f8[:])', cache=True)
def _maxdist(x_i, x_j):
    dist = 0

    leni = len(x_i)
    lenj = len(x_j)

    if leni < lenj:
        n = len(x_i)
    else:
        n = len(x_j)

    for ua in range(n):
        if abs(x_i[ua] - x_j[ua]) > dist:
            dist = abs(x_i[ua] - x_j[ua])

    return dist


@njit('f8(f8[:], f8, i8)', cache=True)
def _compute_sample_entropy(sig, r, m):
    N = sig.shape[0]
    R = r*np.nanstd(sig)
    xlen = N - m

    # Preparing vectors x_B, for calculating denominator
    x_B = np.full((xlen + 1, m), np.inf, dtype='float64')
    for i in range(xlen + 1):
        x_B[i] = sig[i: i + m]

    # Save all matches, compute B
    B = 0
    lenB = len(x_B)
    for i in range(lenB):
        for j in range(i+1, lenB):
            if _maxdist(x_B[i], x_B[j]) <= R:
                B += 1

    # Same for computing nominator A, now with m +=1
    m += 1
    x_A = np.full((N - m + 1, m), np.inf, dtype='float64')
    for i in range(N - m + 1):
        x_A[i] = sig[i: i + m]

    A = 0
    lenA = len(x_A)
    for i in range(lenA):
        for j in range(i+1, lenA):
            if _maxdist(x_A[i], x_A[j]) <= R:
                A += 1

    return -np.log(A / B)

def compute_sample_entropy(sig, r=0.1, m=2):
    """
       Function to compute sample entropy

       Parameters
       ----------
       sig: np.ndarray
           1D signal
       r: np.float64
           filtering threshold, recommended values: 0.1-0.25
       m: int
           window length of compared run of data, recommended (2-8)

       Returns
       -------
       entropy: numpy.float64 (computed as -np.log(A / B))
           approximate entropy

       Example
       -------
       sample_entropy = compute_sample_entropy(data, 0.1, 2)
    """

    # Check the length of the signal with regard to m
    if len(sig) < 100:
        warnings.warn(RuntimeWarning,
                      """The length of the signal is smaller than 400
                       the result might not make sense""")

    return _compute_sample_entropy(sig.astype(float), float(r), int(m))


class SampleEntropy(Method):

    algorithm = 'SAMPLE_ENTROPY'
    algorithm_type = 'univariate'
    version = '1.1.0'
    dtype = [('sampen', 'float32')]

    def __init__(self, **kwargs):
        """
        Sample entropy

        Parameters
        ----------
        sig: np.ndarray
            1D signal
        r: float64
            filtering threshold, recommended values: (0.1-0.25)
        m: int
            window length of compared run of data, recommended (2-8)
        r: float64
            filtering threshold, recommended values: (0.1-0.25)
        """

        super().__init__(compute_sample_entropy, **kwargs)
        self._event_flag = False
