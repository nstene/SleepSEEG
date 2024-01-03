# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


import numpy as np
from multiprocessing import Pool
from scipy.signal import lfilter, remez
from numba import njit
import pandas as pd

# Local imports
from ...utils.method import Method


@njit()
def calculate_background(_win_len, n_samples,
                         iter_back_est, movin_back,
                         half_duration, RMS_signal, _threshold):
    """
    Numba implementation of iterative background estimator
    """

    for jj in range(_win_len, n_samples):

        iter_back_est[jj, :] = min(RMS_signal[jj - half_duration, :][0],
                                   _threshold
                                   * movin_back[jj - half_duration, :][0])

        movin_back[jj, :] = (movin_back[jj - 1, :]
                             + (iter_back_est[jj, :]
                             - iter_back_est[jj - _win_len, :]) / _win_len)

    return movin_back, iter_back_est


def _band_filter(arg):
    """
    Signal filtering and primary detection of 5s window signal.
    """
    _narrowband_filters = arg[0]
    _freq_band = arg[1]
    _bandpass_signal = arg[2]
    _narrowband_filt_order = arg[3]
    _win_len = arg[4]
    _threshold = arg[5]
    _num_samples = arg[6]
    _offset = arg[7]
    _minimum_length = arg[8]
    fs = arg[9]
    signal = arg[10]

    # narrowband signal
    narrowband_signal = lfilter(
        _narrowband_filters, 1,
        np.concatenate([
            _bandpass_signal,
            np.zeros([_narrowband_filt_order // 2, 1])]), axis=0)
    narrowband_signal = np.delete(
        narrowband_signal,
        np.arange(0, _narrowband_filt_order // 2, dtype=int), axis=0)

    # compute RMS (squared) value in 4 cycles window
    half_duration = np.floor(2 / _freq_band).astype('int32')
    four_cycles_duration = (np.ones([1, 2 * half_duration + 1])
                            / (2 * half_duration + 1))
    four_cycles_duration = four_cycles_duration.flatten()

    RMS_signal = np.sqrt(
        np.concatenate([lfilter(four_cycles_duration, 1,
                                narrowband_signal ** 2, axis=0),
                        np.zeros([half_duration, 1])]))
    RMS_signal = np.delete(RMS_signal,
                           np.arange(0, half_duration, dtype=int),
                           axis=0)

    # compute RMS of background (moving window)
    movin_back = RMS_signal.copy()
    movin_back[:_win_len, :] = np.tile(np.mean(RMS_signal[:_win_len, :],
                                               axis=0),
                                       (_win_len, 1))

    iter_back_est = RMS_signal.copy()

    movin_back, iter_back_est = calculate_background(_win_len, _num_samples,
                                                     iter_back_est, movin_back,
                                                     half_duration, RMS_signal,
                                                     _threshold)
    # first detection
    candidate_detections = RMS_signal > _threshold * movin_back

    # find beginning and end of detections
    candidate_detections[0:_offset, :] = False
    candidate_detections[-1 - _offset:, :] = False

    # keep only detections of enough length
    a = candidate_detections.copy().T[0]
    cc = (np.maximum.accumulate((a == 1).cumsum()) -
          np.maximum.accumulate(((a == 1).cumsum() * (a == 0))).astype(float))
    idx = np.where(np.diff(cc) > 0)[0]
    cc[idx] = np.nan

    df = pd.DataFrame(cc)
    df = df.bfill()
    out = np.array(df)
    out[np.where(np.diff(out.T[0]) > 0)[0]+1] = 0

    out[np.where(out.T[0] < _minimum_length)] = 0
    out[np.where(out.T[0] > 1)] = 1
    candidate_detections = out.astype('float32')

    # get event characteristics
    indexes = np.argwhere(np.logical_xor(candidate_detections[:-1, :],
                                         candidate_detections[1:, :]))
    arr_sorted = indexes[indexes[:, 1].argsort(kind='mergesort')]

    index_sample, index_channels = arr_sorted[:, 0], arr_sorted[:, 1]
    event_start = index_sample[::2]
    event_end = index_sample[1::2]
    event_length = event_end - event_start
    detection_channel = index_channels[::2]
    number_of_events = len(event_start)
    values = np.ones([number_of_events, 7])

    for jj in range(0, number_of_events):
        values[jj, :] = [
            max(RMS_signal[event_start[jj]:event_end[jj],
                           detection_channel[jj]] /
                movin_back[event_start[jj]:event_end[jj],
                           detection_channel[jj]]),
            max(RMS_signal[event_start[jj]:event_end[jj],
                           detection_channel[jj]]),
            max(abs(narrowband_signal[event_start[jj]:event_end[jj],
                                      detection_channel[jj]])),
            max(abs(_bandpass_signal[event_start[jj]:event_end[jj],
                                     detection_channel[jj]])),
            max(narrowband_signal[event_start[jj]:event_end[jj],
                                  detection_channel[jj]]),
            min(narrowband_signal[event_start[jj]:event_end[jj],
                                  detection_channel[jj]]),
            max(abs(np.diff(signal[event_start[jj]:event_end[jj],
                                   detection_channel[jj]])))
            ]

    detected_events = np.concatenate([np.expand_dims(detection_channel,
                                                     axis=1),
                                      (fs
                                       * _freq_band
                                       * np.ones([number_of_events, 1])),
                                      np.expand_dims(event_start, axis=1),
                                      np.expand_dims(event_length, axis=1),
                                      values], axis=1)

    return detected_events


def detect_hfo_nicolas(sig, fs=None, mp=1, threshold=3):
    """
    Detection algorithm from Nicolas von Ellenreider.

    Parameters
    ----------
    sig: np.ndarray
        1D raw signal
    fs: float
        Sampling frequency
    mp: int
        Number of processors to use (recommended to use 1 processor and
                                     implement multiprocessing in run_windowed
                                     function)
    threshold: float
        RMS of background times this value

    Returns
    -------
    output: list
        List of tuples with the following structure of detections:
        (event_start, event_stop, osc_frequency)
    """

    sig = np.expand_dims(sig, axis=1)

    # ALGORITHM PARAMETERS
    # background moving window length [seconds]
    window_length = 5

    # INITIALIZATIONS
    num_samples = len(sig)
    window_length = np.round(window_length * fs).astype('int32')

    if num_samples < window_length or window_length == num_samples - 3:
        raise ValueError

    minimum_separation = 0.008  # minimum separation between events [s]
    minimum_separation = np.round(minimum_separation * fs)
    minimum_separation_ripple = 0.02  # minimum separation between events [s]
    minimum_separation_ripple = np.round(minimum_separation_ripple * fs)

    bandpass_filt_order = 201
    narrowband_filt_order = 509

    bandpass_filt = remez(numtaps=bandpass_filt_order,
                          bands=np.array([0, 65, 80, min(0.45 * fs, 500),
                                          min(0.45 * fs + 15, 515), 0.5 * fs]),
                          desired=np.array([0, 1, 0]),
                          grid_density=(bandpass_filt_order + 1) * 16,
                          weight=np.array([10, 1, 1]),
                          fs=fs)

    freq_band = np.array((80, 90, 105, 120, 140, 160, 185, 215,
                          250, 285, 330, 380, 435, 500)) / fs
    num_bands = len(freq_band) - 1
    narrowband_filters = np.zeros((num_bands, narrowband_filt_order))

    for ii in range(0, num_bands):
        narrowband_filters[ii:] = remez(numtaps=narrowband_filt_order,
                                        bands=np.array([
                                            0, freq_band[ii] - 7 / fs,
                                            freq_band[ii] + 3 / fs,
                                            freq_band[ii + 1] - 3 / fs,
                                            freq_band[ii + 1] + 7 / fs, 0.5]),
                                        desired=np.array([0, 1, 0]),
                                        weight=np.array([10, 1, 1]))

    freq_band = (freq_band[0:-1] + freq_band[1:]) / 2

    minimum_length = np.round(
        np.sqrt(
            np.sum(
                np.tile(
                    np.arange(-narrowband_filt_order // 2,
                              narrowband_filt_order // 2) ** 2,
                    [num_bands, 1]) * narrowband_filters ** 2, 1)
            / np.sum(narrowband_filters ** 2, 1)) + 4 / freq_band)
    minimum_length = minimum_length.astype('int32').tolist()

    offset = int((narrowband_filt_order + bandpass_filt_order) / 2 - 1)

    ar = np.concatenate([sig, np.zeros([bandpass_filt_order // 2, 1])])
    bandpass_signal = lfilter(bandpass_filt, 1, ar, axis=0)
    bandpass_signal = np.delete(bandpass_signal,
                                np.arange(0,
                                          bandpass_filt_order // 2,
                                          dtype=int), axis=0)

    x = [[narrowband_filters[ii, :], freq_band[ii], bandpass_signal,
          narrowband_filt_order, window_length, threshold, num_samples,
          offset, minimum_length[ii], fs, sig] for ii in range(0, num_bands)]

    # multiprocessing of signal filtering and primary detections
    if mp > 1:
        pool = Pool()
        results = pool.map(_band_filter, x)

    else:
        results = []
        for arg in x:
            result = _band_filter(arg)
            results.append(result)

    events = np.array([array for arrays in results for array in arrays])
    ev = events.copy()

    event_counter = events.shape[0]

    if event_counter > 0:
        # separate in ripples and fast ripples, join overlapping events
        ev = np.concatenate([ev[:, [0, 1, 2, 3, 4, 8, 9, 10]],
                             np.ones([ev.shape[0], 1])], axis=1)
        ripples = np.zeros([event_counter, 9])
        fast_ripples = ripples.copy()

        ripple_counter = 0
        fast_ripple_counter = 0

        # ripples
        events_in_channel = ev[(np.where(ev[:, 1] < 250))[0].tolist(), :]
        channel_event = np.zeros([events_in_channel.shape[0], 9])
        counter = 0

        while events_in_channel.shape[0] > 0:
            N = events_in_channel.shape[0]
            all_events = (events_in_channel
                          if counter == 0
                          else np.concatenate([
                              events_in_channel,
                              channel_event[:counter + 1, :]], axis=0))
            counter += 1

            index_of_events = np.logical_and(
                (all_events[:, 2] + all_events[:, 3]
                 + minimum_separation_ripple > events_in_channel[0, 2]),
                (all_events[:, 2] < events_in_channel[0, 2]
                 + events_in_channel[0, 3] + minimum_separation_ripple))

            overlapping_events = all_events[index_of_events, :]
            events_in_channel = np.delete(events_in_channel,
                                          index_of_events[0:N],
                                          axis=0)

            idx_delete = np.where(index_of_events[N:])[0].tolist()

            channel_event = (channel_event
                             if len(index_of_events[N + 1:]) == 0
                             else np.delete(channel_event, idx_delete, axis=0))
            counter = counter - np.count_nonzero(index_of_events[N:])

            channel_event[counter - 1, :] = [
                round(overlapping_events[0, 0]),
                float(np.mean(overlapping_events[:, 1])),
                round(min(overlapping_events[:, 2])),
                round(max(overlapping_events[:, 2]
                          + overlapping_events[:, 3])
                      - min(overlapping_events[:, 2])),
                float(max(overlapping_events[:, 4])),
                float(max(overlapping_events[:, 5])),
                float(min(overlapping_events[:, 6])),
                round(max(max(overlapping_events[:, 7]), 1)),
                round(sum(overlapping_events[:, 8]))
                ]

        channel_event = channel_event[:counter, :]
        ripples[(ripple_counter
                 + np.arange(0
                             + channel_event.shape[0])).tolist(),
                :] = channel_event
        ripple_counter += channel_event.shape[0]

        # fast ripples
        events_in_channel = ev[(np.where(ev[:, 1] > 250))[0].tolist(), :]
        channel_event = np.zeros([events_in_channel.shape[0], 9])
        counter = 0

        while events_in_channel.shape[0] > 0:
            N = events_in_channel.shape[0]
            all_events = (events_in_channel
                          if counter == 0
                          else np.concatenate([
                              events_in_channel,
                              channel_event[:counter + 1, :]], axis=0))
            counter += 1

            index_of_events = np.logical_and(
                (all_events[:, 2] + all_events[:, 3]
                 + minimum_separation > events_in_channel[0, 2]),
                (all_events[:, 2] < events_in_channel[0, 2]
                 + events_in_channel[0, 3] + minimum_separation))

            overlapping_events = all_events[index_of_events, :]
            events_in_channel = np.delete(events_in_channel,
                                          index_of_events[0:N],
                                          axis=0)

            idx_delete = np.where(index_of_events[N:])[0].tolist()

            channel_event = (channel_event
                             if len(index_of_events[N + 1:]) == 0
                             else np.delete(channel_event, idx_delete, axis=0))
            counter = counter - np.count_nonzero(index_of_events[N:])

            channel_event[counter - 1, :] = [
                round(overlapping_events[0, 0]),
                float(np.mean(overlapping_events[:, 1])),
                round(min(overlapping_events[:, 2])),
                round(max(overlapping_events[:, 2]
                          + overlapping_events[:, 3])
                      - min(overlapping_events[:, 2])),
                float(max(overlapping_events[:, 4])),
                float(max(overlapping_events[:, 5])),
                float(min(overlapping_events[:, 6])),
                round(max(max(overlapping_events[:, 7]), 1)),
                round(sum(overlapping_events[:, 8]))
                                             ]

        channel_event = channel_event[:counter, :]
        fast_ripples[(fast_ripple_counter
                      + np.arange(0
                                  + channel_event.shape[0])).tolist(),
                     :] = channel_event
        fast_ripple_counter += channel_event.shape[0]

        ripples = ripples[0:ripple_counter, :]
        fast_ripples = fast_ripples[0:fast_ripple_counter, :]

        oscillations = np.concatenate([ripples, fast_ripples], axis=0)
        output = [tuple([int(oscillations[i, 2]),
                         int(oscillations[i, 2] +
                             oscillations[i, 3]), oscillations[i, 1]])
                  for i in np.arange(np.shape(oscillations)[0])]
    else:
        output = []

    return output


class NicolasDetector(Method):

    algorithm = 'NICOLAS_DETECTOR'
    version = '1.0.0'
    algorithm_type = 'event'
    dtype = [('event_start', 'int32'),
             ('event_stop', 'int32'),
             ('frequency', 'float32')]

    def __init__(self, **kwargs):
        """
        Detection algorithm from Nicolas von Ellenreider.

        Parameters
        ----------
        fs: int
            Sampling frequency
        window_size: int
            Sliding window size in samples.
            !!! Algorithm needs at least 5s window!!!!!
        window_overlap: float
            Fraction of the window overlap (0 to 1)
            !!!! Algorithm is designed with no overlap!!!
        sample_offset: int
            Offset which is added to the final detection. This is used when the
            function is run in separate windows. Default = 0

        References
        ----------
        von Ellenrieder, N., Andrade-Valença, L.P., Dubeau, F., Gotman, J.,
        2012. Automatic detection of fast oscillations (40–200 Hz) in scalp EEG
        recordings. Clin. Neurophysiol. 123, 670–680.

        von Ellenrieder, N., Frauscher, B., Dubeau, F., Gotman, J., 2016.
        Interaction with slow waves during sleep improves discrimination of
        physiological and pathological high frequency oscillations (80–500 Hz).
        Epilepsia 57, 869–878.
        """

        super().__init__(detect_hfo_nicolas, **kwargs)
        self._event_flag = True
