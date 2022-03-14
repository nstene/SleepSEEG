# Third party imports
import numpy as np
import scipy.signal as sp

# Local imports
from ..utils.method import Method

def compute_low_f_marker(signal):
    """
    Function to compute power ratio of two signal windows filtered on different Frequencies based on Lundstrom et al. 2021
    https://www.medrxiv.org/content/10.1101/2021.06.04.21258382v1.full.pdf

    Parameters
    ----------
    infra_signal: np.array
        infra signal window filtered between 0.02-0.5 Hz (order 1 butterworth filter 
        recommended for this use)
    main_signal: np.array
        main signal window filtered between 2-4 Hz for the first marker and 20-50 Hz for
        the second marker (order 1 butterworth filter recommended for this use)

    Returns
    --------
    low_f_marker: float32
        returns median of given time window 
    """
    order = 1
    fs = 5000

    lowband=[0.02, 0.5]
    highband=[2.0, 4.0]

    nyq = fs * 0.5

    lowband = np.divide(lowband, nyq)
    highband = np.divide(highband, nyq)

    [b, a] = sp.butter(order, lowband, btype='bandpass', analog=False)
    infra_signal = sp.filtfilt(b, a, signal, axis=0)

    [b, a] = sp.butter(order, highband, btype='bandpass', analog=False)
    main_signal = sp.filtfilt(b, a, signal, axis=0)

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
        infra_signal: np.array
            infra signal window filtered between 0.02-0.5 Hz (order 1 butterworth filter 
            recommended for this use)
        main_signal: np.array
            main signal window filtered between 2-4 Hz for the first marker and 20-50 Hz for
            the second marker (order 1 butterworth filter recommended for this use)
        """

        super().__init__(compute_low_f_marker, **kwargs)