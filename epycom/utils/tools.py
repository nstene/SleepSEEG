# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third party imports
import numpy as np

# Local imports


def try_jit_decorate(jit_kwargs):
    try:
        from numba import jit
        return jit(**jit_kwargs)
    except ImportError:
        return lambda x: x


def try_njit_decorate(jit_args, jit_kwargs):
    try:
        from numba import njit
        return njit(jit_args, **jit_kwargs)
    except ImportError:
        return lambda x: x


def validate_signal(sig):
    # Check if the signal is flat
    if not np.any(np.diff(sig)):
        return True

    # Check if signal contains NaNs
    if np.any(np.isnan(sig)):
        return True

    return False
