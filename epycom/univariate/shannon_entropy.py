# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
from scipy import stats as stats
import numpy as np

# Local imports
from ..utils.method import Method


def compute_shanon_entropy(sig):
    """
    Fucntion computes shannon entropy of given signal

    Parameters
    ----------
    sig: np.ndarray
        Signal to analyze

    Returns
    -------
    entro: np.float64
        Computed Shannon entropy of given signal
    """
    nbins = 10
    counts = np.histogram(sig,nbins)
    entro = stats.entropy(counts[0], base=2)               # shan_en = -sum(p(xi)*log(p(xi)))
    return entro


class ShannonEntropy(Method):

    algorithm = 'SHANNON_ENTROPY'
    algorithm_type = 'univariate'
    version = '2.0.0'
    dtype = [('shannon', 'float32')]

    def __init__(self, **kwargs):
        """
        Shannon entropy

        Parameters
        ----------
        sig: np.ndarray
            Signal to analyze
        """

        super().__init__(compute_shanon_entropy, **kwargs)
        self._event_flag = False

