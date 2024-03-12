# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Third party imports
import numpy as np

# Local imports
from ..utils.method import Method
from epycom.univariate.sample_entropy import compute_sample_entropy

def compute_multiscale_entropy(sig, r=0.1, m=2, min_scale=2, max_scale=2):
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
        min_scale: unsigned int
            minimum number of samples to collapse
        max_scale: unsigned int
            maximum number of samples to collapse

        Returns
        -------
        entropy: numpy.float64 (computed as -np.log(A / B))
            approximate entropy

        Example
        -------
        sample_entropy = compute_sample_entropy(data, 0.1, 2)
    """

    # Input checks
    if not isinstance(min_scale, int) or min_scale <= 0:
        msg = f"""
        min_scale has to be unsigned integer. Not {type(min_scale)}, {min_scale}
        """
        raise ValueError(msg)

    if not isinstance(max_scale, int) or max_scale <= 0:
        msg = f"""
        max_cale has to be unsigned integer. Not {type(max_scale)}, {max_scale}
        """
        raise ValueError(msg)

    if min_scale > max_scale:
        msg = f"""
        min_scale has to be smaller than max_scale. Not {min_scale}, {max_scale}
        """
        raise ValueError(msg)

    sm_entropy_vals = np.empty(max_scale - min_scale + 1)
    for i, scale in enumerate(range(min_scale, max_scale+1)):

        reshaped_arr = np.resize(sig, (len(sig) // scale, scale))
        new_sig = np.mean(reshaped_arr, axis=1)

        sm_entropy_vals[i] = compute_sample_entropy(new_sig.astype(float),
                                                    float(r),
                                                    int(m))

    max_entropy_scale = np.argmax(sm_entropy_vals) + min_scale

    return np.sum(sm_entropy_vals), max_entropy_scale


class MultiscaleEntropy(Method):

    algorithm = 'MULTISCALE_ENTROPY'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('me_sum', 'float32'),
             ('me_maxscale', 'uint8')]

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
        min_scale: unsigned int
            minimum number of samples to collapse
        max_scale: unsigned int
            maximum number of samples to collapse
        """

        super().__init__(compute_multiscale_entropy, **kwargs)
        self._event_flag = False
