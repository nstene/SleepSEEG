# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
from scipy.stats import entropy

# Local imports
from ..utils.method import Method

def compute_selection_entropy(sig, nbins = 10):
    """
    Calculation of Selection Entropy:
    selection entropy of sig[0] with respect to sig[1]
    SelEn = entropy(H1) - entropy(H2) + sum(pxx*np.log(pxx))

    Parameters
    ----------
    sig: np.array
        2D numpy array of shape (signals, samples), time series (int, float)

    Returns
    ---------
    SelEn: float
        Value of selection entropy between sig[0] and sig[1]

    Example
    -------
    SelEn = compute_selection_entropy(sig)

    References
    ----------
    Selection entropy: The information hidden within neuronal patterns, Fagerholm (2023)
        https://doi.org/10.1103/PhysRevResearch.5.023197
    """

    if type(sig) != np.ndarray:
        raise TypeError("Signals have to be in numpy arrays!")

    if sig.ndim != 2:
        raise TypeError(f"The array must have two dimensions not {sig.ndim}!")

    h1 = np.histogram(sig[0], bins=nbins)
    h2 = np.histogram(sig[1], bins=nbins)
    L = np.sum(h1[0])
    H1 = (h1[0]) /L
    H2 = (h2[0]) /L

    pm = np.minimum(H1, H2)
    pn = np.maximum(H1, H2)
    x1 = np.where(pn != pm)[0] 
    #for pn=pm we would calculate log2(0), but those cancels out
    x2 = np.where((pn != pm) & (pm != 0))[0]
    #for x->0: x*log2(x) -> 0 so we can ignore those indexes

    SelEn = (np.sum(pm[x2]*np.log2(pn[x2]/pm[x2]-1))
    - np.sum(pn[x1]*np.log2(1-pm[x1]/pn[x1])))
    # the change from original proposed algorithm is change in order of sums, 
    # that is possible because the sum is finite
    return SelEn


class SelectionEntropy(Method):

    algorithm = 'SELECTION_ENTROPY'
    algorithm_type = 'bivariate'
    version = '1.0.0'
    dtype = [('SelEn', 'float32')]

    def __init__(self, **kwargs):
        """
        Calculation of Selection Entropy:
        selection entropy between sig[0] and sig[1]
        """

        super().__init__(compute_selection_entropy, **kwargs)
