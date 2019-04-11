# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


from scipy.signal import butter, filtfilt

from ..feature_extraction import extract_line_lenght
from ..thresholds import th_std
from ..io.data_operations import create_output_df

# %% Line-length detector


def detect_hfo_ll(data, fs, low_fc, high_fc,
                  threshold, window_size, window_overlap):
    """
    Line-length detection algorithm {Gardner et al. 2007, Worrell et al. 2018,
    Akiyama et al. 2011}.

    Parameters
    ----------
    data: numpy array
        1D array with raw data
    fs: int
        Sampling frequency
    low_fc: float
        Low cut-off frequency
    high_fc: float
        High cut-off frequency
    window_size: float
        Sliding window size in seconds
    window_overlap: float
        Fraction of the window overlap (0 to 1)

    Returns:
    --------
    df_out: pandas dataframe
        Output dataframe with detections
    """

    # Calculate window values for easier operation
    samp_win_size = int(window_size * fs)
    samp_win_inc = int(samp_win_size * window_overlap)

    # Create output dataframe

    df_out = create_output_df()

    # Filter the signal

    b, a = butter(3, [low_fc / (fs / 2), high_fc / (fs / 2)], 'bandpass')
    filt_data = filtfilt(b, a, data)

    # Transform the signal - one sample window shift

    #LL = compute_line_lenght(filt_data, window_size*fs)

    # Alternative approach - overlapping window

    win_start = 0
    win_stop = window_size * fs
    LL = []
    while win_start < len(filt_data):
        if win_stop > len(filt_data):
            win_stop = len(filt_data)

        LL.append(extract_line_lenght(filt_data[int(win_start):int(win_stop)]))

        win_start += samp_win_inc
        win_stop += samp_win_inc

    # Create threshold
    det_th = th_std(LL, threshold)

    # Detect
    LL_idx = 0
    df_idx = 0
    while LL_idx < len(LL):
        if LL[LL_idx] >= det_th:
            event_start = LL_idx * samp_win_inc
            while LL_idx < len(LL) and LL[LL_idx] >= det_th:
                LL_idx += 1
            event_stop = (LL_idx * samp_win_inc) + samp_win_size

            if event_stop > len(data):
                event_stop = len(data)

            # Optional feature calculations can go here

            # Write into dataframe
            df_out.loc[df_idx] = [event_start, event_stop]
            df_idx += 1

            LL_idx += 1
        else:
            LL_idx += 1

    return df_out
