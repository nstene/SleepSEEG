# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Third party imports
import numpy as np

# Local imports
from ..utils.method import Method
from epycom.univariate import (Sample_Entropy)

def compute_multiscale_entropy(sig, r=0.1, m=2, scale = 2):
    """
       Function to compute multiscale entropy

       Parameters
       ----------
       sig: np.ndarray
           1D signal
       r: np.float64
           filtering threshold, recommended values: 0.1-0.25
       m: int
           window length of compared run of data, recommended (2-8)
       scale: insigned int
            number of samples to collapse

       Returns
       -------
       entropy: numpy.float64 (computed as -np.log(A / B))
           approximate entropy

       Example
       -------
       sample_entropy = compute_sample_entropy(data, 0.1, 2)
    """
    if not isinstance(scale, int) or scale <= 0:
        msg = "Scale has to be unsigned integer. Not {}"
        raise ValueError(msg.format(scale))
    scale = int(scale)
    
    len_new_sig = len(sig)//scale
    new_sig = np.zeros(len_new_sig)

    # compute new signal as nonoverlaping averages in the signal elements of 
    # length scale
    for j in range (len_new_sig):
        for i in range (scale):
            new_sig[j] += sig[(j)*scale+i]
        new_sig[j] /= scale
    print(len(new_sig))
    
    # compute sample entropy of the new signal
    return Sample_Entropy()._compute_sample_entropy(new_sig.astype(float), float(r), int(m))


class MultiscaleEntropy(Method):

    algorithm = 'MULTISCALE_ENTROPY'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('multen', 'float32')]

    def __init__(self, **kwargs):
        """
        Sample entropy

        Parameters
        ----------
        sig: np.ndarray
            1D signal
        m: int
            window length of compared run of data, recommended (2-8)
        r: float64
            filtering threshold, recommended values: (0.1-0.25)
        scale: unsigned int
            number of samples to colapse        
        """

        super().__init__(compute_multiscale_entropy, **kwargs)
        self._event_flag = False
