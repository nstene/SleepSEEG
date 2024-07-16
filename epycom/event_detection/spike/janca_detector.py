# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
import sys
import math
from fractions import Fraction

# Third pary imports
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.signal import (decimate, filtfilt, buttord, butter, freqz, cheb2ord,
                          cheby2, hilbert)
from scipy.interpolate import interp1d

# Local imports
from ...utils.method import Method


def detect_spikes_janca(sig, fs=None,
                        bandwidth=(10, 60), line_freq=50,
                        k1=3.65, k2=3.65, k3=0,
                        polyspike_union_time=0.12,
                        decimation=200,
                        buffering=300,
                        discharge_tol=0.005):
    """
    Simplified python version of janca's EEG spike detector.
    {Janca et al. 2015} Reprogrammed from Matlab version by Daniel Uher.
    Modified by Jan Cimbalnik.

    Parameters
    ----------
    sig: np.ndarray
        1D numpy array of EEG data - NOT TRUE!!! check!!
    fs: int
        sampling frequency of the signal
    bandwidth: list
        low and high filtering boundaries (Hz)
    line_freq: int
        power line frequency to be filtered out (Hz). Default=50
    k1: float
        threshold value for obvious spikes. Default=3.65
    k2: float
        defines ambiguous spike threshold. Ambiguous is accepted when
        simultaneous obvious detection is in other channel k1>=k2. Default=k1
    k3: float
        decrease the threshold value k1*(mode+median)-k3(mean-mode). Default=0
    decimation: int
        decimation by number of samples for decimated signal. Default=200
    buffering: int
        size of buffer. Default=300
    polyspike_union_time: float
        Time to unify spikes into one detection on one channel. Default=0.12
    discharge_tol: float
        Time between spikes. Default=0.005
    """

    fs = int(fs)
    orig_fs = fs
    orig_winsize = 5*fs
    orig_noverlap = 4*fs

    n_samples = len(sig)

    N_seg = np.floor(n_samples/(buffering*fs))
    if N_seg < 1:
        N_seg = 1

    T_seg = round(n_samples/N_seg/fs)

    # indices of segments with two-side overlap
    index_start = np.arange(0, n_samples, T_seg*fs, dtype=int)

    if len(index_start) > 1:
        index_start[1:] = index_start[1:] - (3*orig_winsize)

        index_stop = index_start + T_seg*fs + 2*(3*orig_winsize)-1
        index_stop[0] = index_stop[0] - (3*orig_winsize)
        index_stop[-1] = n_samples-1

        if index_stop[-1]-index_start[-1] < T_seg*fs:
            index_start = np.delete(index_start, -1)
            index_stop = np.delete(index_stop, -2)

    else:
        index_stop = np.array([n_samples])

    out_list = []

    for buff_i in range(0, len(index_stop)):

        fs = orig_fs
        winsize = orig_winsize
        noverlap = orig_noverlap

        RFactor = fs/decimation
        d = sig[int(index_start[buff_i]):int(index_stop[buff_i])].copy()
        if RFactor > 1 or decimation != orig_fs:
            winsize = winsize/fs
            noverlap = noverlap/fs

            decItterations = int(np.ceil(np.log10(RFactor)))

            for i in range(0, decItterations):
                if i == int(decItterations-1):
                    fs_out = decimation
                else:
                    fs_out = round(fs/(RFactor**(1/decItterations)))

                p = Fraction(fs_out, int(fs))
                q = p.denominator
                p = p.numerator

                sub_d_res = decimate(d, int(q/p),
                                     n=2*100*max(p, q), ftype='fir')

                d = sub_d_res
                fs = fs_out

            winsize = round(winsize*fs)
            noverlap = round(noverlap*fs)

        if noverlap < 1:
            index = np.arange(0, d.shape[0]-winsize+1,
                              round(winsize*(1-noverlap)))
        else:
            index = np.arange(0, d.shape[0]-winsize+1,
                              winsize-noverlap)

        """-- 50/60 Hz filterring ------------------------------------------"""
        d = _filt50Hz(d, fs, bandwidth)
        [bb, aa] = butter(2, 2*1/fs, 'highpass')
        d_decim = filtfilt(bb, aa, d, axis=0, padtype='odd',
                           padlen=3*(max(len(bb), len(aa))-1))

        """-- bandwidth filtering  -----------------------------------------"""

        if decimation == 200:
            # lowpass
            Wp = 2*bandwidth[1]/fs
            Ws = 2*bandwidth[1]/fs+0.1
            Rp = 6
            Rs = 60
            n, _ = cheb2ord(Wp, Ws, Rp, Rs)
            bl, al = cheby2(n, Rs, Ws)

            # high pass
            Wp = 2*bandwidth[0]/fs
            Ws = 2*bandwidth[0]/fs-0.05
            Rp = 6
            Rs = 60
            n, _ = cheb2ord(Wp, Ws, Rp, Rs)
            bh, ah = cheby2(n, Rs, Ws, 'highpass')
        else:
            # lowpass
            Wp = 2*bandwidth[1]/fs
            Ws = 2*bandwidth[1]/fs + 0.1
            if Ws > 1:
                Ws = 1
            Rp = 1
            Rs = 60
            but_order, Ws = buttord(Wp, Ws, Rp, Rs)
            bl, al = butter(but_order, Ws)

            # high pass
            Wp = 2*bandwidth[0]/fs
            Ws = 2*bandwidth[0]/fs - 0.05
            if Ws < 0:
                Ws = 0.1
            Rp = 6
            Rs = 60
            but_order, Ws = buttord(Wp, Ws, Rp, Rs)
            bh, ah = butter(but_order, Ws, 'high')

        _, hl = freqz(bl, al, worN=10*fs, fs=fs)
        _, hh = freqz(bh, ah, worN=10*fs, fs=fs)
        if max(abs(hl)) > 1.001 or max(abs(hh)) > 1.001:
            sys.exit('filters are probably unstable !!!')

        d = filtfilt(bh, ah, d, axis=0, padtype='odd',
                     padlen=3*(max(len(bh), len(ah))-1))
        if bandwidth[1] != fs/2:
            d = filtfilt(bl, al, d, axis=0, padtype='odd',
                         padlen=3*(max(len(bl), len(al))-1))

        envelope = abs(hilbert(d))
        phat = np.zeros([len(index), 2])

        for k in range(0, index.shape[0]):  # for each segment

            segment = envelope[index[k]:index[k]+winsize]
            segment = segment[segment > 0]

            # estimation of segment's distribution using MLE
            phat[k, 0] = np.mean(np.log(segment))
            phat[k, 1] = np.std(np.log(segment))

        r = len(envelope)/len(index)
        n_average = winsize/fs

        if round(n_average*(fs/r)) > 1:
            phat = filtfilt(np.ones(
                round(n_average*(fs/r)))/(round(n_average*(fs/r))),
                1, phat, axis=0)

        # interpolation of thresholds value to threshold curve (like backround)
        phat_int = []
        prah_int = []
        if phat.shape[0] > 1:
            f = interp1d(index+round(winsize/2), phat[:, 0], 'cubic')
            interp_x = np.arange(index[0], index[-1]+1, 1)+round(winsize/2)
            phat_int = np.zeros((interp_x.shape[0], 2))
            phat_int[:, 0] = f(interp_x)

            f = interp1d(index+round(winsize/2), phat[:, 1], 'cubic')
            interp_x = np.arange(index[0], index[-1]+1, 1)+round(winsize/2)
            phat_int[:, 1] = f(interp_x)

            top = np.ones((interp_x[0]-1, 2))*phat_int[0, :]
            bot = np.ones((envelope.shape[0]-interp_x[-1], 2))*phat_int[-1, :]
            phat_int = np.concatenate((top, phat_int, bot))

        else:
            phat_int = np.multiply(phat, np.ones(d.shape[0], 2))

        lognormal_mode = np.exp(phat_int[:, 0]-np.power(phat_int[:, 1], 2))
        lognormal_median = np.exp(phat_int[:, 0])
        lognormal_mean = np.exp(phat_int[:, 0]+(np.power(phat_int[:, 1], 2)/2))

        prah_int = np.zeros((phat_int.shape))
        prah_int[:, 0] = (k1 * (lognormal_mode+lognormal_median)
                          - k3 * (lognormal_mean-lognormal_mode))
        if k2 != k1:
            prah_int[:, 1] = (k2 * (lognormal_mode+lognormal_median)
                              - k3 * (lognormal_mean-lognormal_mode))

        # compute CDF of lognormal distribution
        a = np.log(envelope) - phat_int[:, 0]
        b = np.sqrt(2*np.power(phat_int[:, 1], 2))
        envelope_cdf = 0.5 + 0.5*np.asarray([math.erf(x) for x in a / b])

        # compute PDF of lognormal distribution
        z = np.sqrt(2*math.pi)
        x = np.multiply(envelope, phat_int[:, 1] * z)
        y = np.log(envelope) - phat_int[:, 0]
        envelope_pdf = np.exp(-0.5*np.power(np.divide(y, phat_int[:, 1]), 2))
        envelope_pdf = envelope_pdf / x

        """ detection of obvious and ambiguous spike """
        markers_high = _local_maxima_detection(envelope, prah_int[:, 0], fs,
                                               polyspike_union_time)

        markers_high = _detection_union(markers_high, envelope,
                                        polyspike_union_time*fs)

        if k2 != k1:
            markers_low = _local_maxima_detection(envelope, prah_int[:, 1], fs,
                                                  polyspike_union_time)
            markers_low = _detection_union(markers_low, envelope,
                                           polyspike_union_time*fs)
        else:
            markers_low = markers_high

        markers_high[:fs] = False
        markers_high[-fs:] = False
        markers_low[:fs] = False
        markers_low[-fs:] = False

        obvious_M = markers_high

        out_dtype = [('event_peak', 'float32'),  # spike position
                     # ('dur', 'float32'),  # spike duration - fix value 5 ms
                     ('condition', 'float32'),  # spike condition
                     # ('weight', 'float32'),  # spike weight "CDF"
                     # ('pdf', 'float32')  # spike probability "PDF"
                     ]

        t_dur = 0.005

        if np.any(markers_high):

            idx = np.nonzero(markers_high)[0]

            sub_out = np.empty(len(idx), dtype=out_dtype)

            sub_out["event_peak"] = idx/fs
            # sub_out["dur"] = t_dur*np.ones((len(idx)))
            sub_out["condition"] = np.ones((len(idx)))
            # sub_out["weight"] = envelope_cdf[idx]
            # sub_out["pdf"] = envelope_pdf[idx]

        else:
            sub_out = np.empty(0, dtype=out_dtype)

        if k2 != k1:
            # ambiguous spike events output
            if np.any(markers_low):

                idx = np.nonzero(markers_low)[0]

                idx[np.nonzero(markers_high[idx])] = False

                amby_out = np.empty(len(idx), dtype=out_dtype)

                for i in range(0, len(idx)):
                    if np.any(obvious_M[np.arange(round(idx[i]-0.01*fs),
                                                  round(idx[i]+0.01*fs))]):

                        amby_out["event_peak"] = idx[i]/fs
                        # amby_out["dur"] = t_dur
                        amby_out["condition"] = 0.5
                        # amby_out["weight"] = envelope_cdf[idx[i]]
                        # amby_out["pdf"] = envelope_pdf[idx[i]]

                sub_out = np.concatenate([sub_out, amby_out])

        # making M stack pointer of events
        M = np.zeros(len(d))
        for k in range(0, len(sub_out["event_peak"])):
            pom = np.arange(
                round(sub_out["event_peak"][k]*fs),
                round(sub_out["event_peak"][k]*fs+discharge_tol*fs)+1, 1)
            M[pom] = sub_out["condition"][k]

        pom1 = np.concatenate((np.array([False]), M > 0))
        pom2 = np.concatenate((M > 0, np.array([False])))
        point1 = np.argwhere(np.diff(np.asarray(pom1, dtype=int)) > 0)
        point2 = np.argwhere(np.diff(np.asarray(pom2, dtype=int)) < 0)
        point = np.vstack((point1.T, point2.T)).T
        point[:, 1] += 1

        discharges_dtype = [
            #  ('MV', 'float32'),  # type 1-obvious,0.5- ambiguous
            ('max_amplitude_env', 'float32'),  # max. amplitude of envelope
            ('event_start', 'float32'),  # event start position
            ('event_duration', 'float32'),  # event duration
            #  ('MW', 'float32'),  # statistical weight
            #  ('MPDF', 'float32'),  # statistical weight
            ('max_amplitude', 'float32')  # amplitude of signal
            ]

        sub_discharges = np.zeros(point.shape[0], dtype=discharges_dtype)
        for k in range(0, point.shape[0]):
            seg = M[point[k, 0]:point[k, 1]]

            mv = np.max(seg)

            seg = (envelope[point[k, 0]:point[k, 1]]
                   - (prah_int[point[k, 0]:point[k, 1], 0]/k1))
            ma = np.max(abs(seg), axis=0)
            seg = d_decim[point[k, 0]:point[k, 1]]
            mraw = np.max(abs(seg))
            poz = np.argmax(abs(seg))

            mraw = mraw * np.sign(seg.flatten(order='F')[poz])

            # CHECKED HERE

            seg = envelope_cdf[point[k, 0]:point[k, 1]]
            mw = np.max(seg)

            seg = envelope_pdf[point[k, 0]:point[k, 1]]
            mpdf = np.max(np.multiply(seg, M[point[k, 0]:point[k, 1]] > 0))

            pom = M[point[k, 0]:point[k, 1]] > 0
            row = np.where(pom)[0]
            mp = row[-1] + point[k, 0]

            sub_discharges["max_amplitude"][k] = mraw
            # sub_discharges["MV"][k] = mv
            sub_discharges["max_amplitude_env"][k] = ma
            # sub_discharges["MW"][k] = mw
            # sub_discharges["MPDF"][k] = mpdf
            sub_discharges["event_duration"][k] = (point[k, 1]-point[k, 0]-1)/fs
            sub_discharges["event_start"][k] = mp/fs

        # conecting of subsections ----------------

        # removing of two side overlap detections
        if len(sub_out) and len(index_stop) > 1:

            start_pos = sub_out["pos"] < (buff_i > 1)*(3*orig_winsize)/orig_fs

            index_diff = index_stop[buff_i]-index_start[buff_i]
            end_pos = (sub_out["pos"] > (index_diff-(buff_i < len(index_stop))
                                         * (3*orig_winsize))/orig_fs)

            idx_evt = (start_pos) | end_pos

            sub_out = sub_out[~idx_evt]
            sub_discharges = sub_discharges[~idx_evt]

        sub_out = rfn.merge_arrays([sub_out, sub_discharges],
                                   flatten=True, usemask=False)
        out_list.append(sub_out)

    out = np.concatenate(out_list)

    # Convert seconds to samples
    out["event_peak"] *= orig_fs
    out["event_duration"] *= orig_fs
    out["event_start"] *= orig_fs

    return out[['event_peak', 'max_amplitude_env', 'max_amplitude']].tolist()


def _local_maxima_detection(envelope, prah_int, fs, polyspike_union_time):

    envelope_flat = envelope.flatten('C')
    marker1 = (envelope_flat > prah_int).astype(int)
    non_zeros_1 = np.argwhere(np.diff(np.concatenate(([0], marker1))) > 0)
    non_zeros_2 = np.argwhere(np.diff(np.concatenate((marker1, [0]))) < 0)
    if non_zeros_1.size and non_zeros_2.size:
        point = np.vstack((non_zeros_1.T, non_zeros_2.T)).T
    else:
        point = np.array([])

    marker1 = np.full(envelope.shape, False)
    for k in range(0, point.shape[0]):
        # detection of local maxima in section which crossed threshold curve
        if ((point[k, 1]+1)-point[k, 0]) > 2:
            seg = envelope[point[k, 0]:point[k, 1]+1]
            seg_s = np.diff(seg)
            seg_s = np.sign(seg_s)
            seg_s_nz = (np.diff(np.concatenate(([0], seg_s))) < 0).astype(int)
            seg_s = np.nonzero(seg_s_nz)[0]  # local maxima
            marker1[point[k, 0]+seg_s] = True

        elif (point[k, 1]+1)-point[k, 0] <= 2:
            seg = envelope[point[k, 0]:point[k, 1]+1]
            s_max = [i for i, j in enumerate(seg) if j == max(seg)]  # maxima
            marker1[point[k, 0] + s_max] = True

    # union of section, where local maxima are close together
    # < (1/f_low + 0.02 sec.)~ 120 ms
    pointer = np.where(marker1)[0]  # index of local maxima
    state_previous = False
    start = 0

    for k in range(0, max(pointer.shape)):
        if np.ceil(pointer[k]+polyspike_union_time*fs) > marker1.shape[0]:
            seg = marker1[pointer[k]+1:]
        else:
            seg = marker1[pointer[k]+1:
                          int(np.ceil(pointer[k]+polyspike_union_time*fs)+1)]

        if state_previous:
            if np.sum(seg) > 0:
                state_previous = True
            else:
                state_previous = False
                marker1[start:pointer[k]] = True

        else:
            if np.sum(seg) > 0:
                state_previous = True
                start = pointer[k]

    # finding of the highes maxima of the section with local maxima
    non_zeros_1 = np.argwhere(np.diff(np.concatenate(([0], marker1))) > 0)
    non_zeros_2 = np.argwhere(np.diff(np.concatenate((marker1, [0]))) < 0)+1
    if non_zeros_1.size and non_zeros_2.size:
        point = np.vstack((non_zeros_1.T, non_zeros_2.T)).T
    else:
        point = np.array([])

    # local maxima with gradient in souroundings
    for k in range(0, point.shape[0]):

        if point[k, 1]-point[k, 0] > 2:
            local_m = pointer[(pointer >= point[k, 0])
                              & (pointer <= point[k, 1])]
            marker1[point[k, 0]:point[k, 1]] = False
            local_max_val = envelope[local_m]  # envelope in local maxima
            pom = np.concatenate([np.array([0]), local_max_val, np.array([0])])
            pom = np.sign(np.diff(pom))
            pom = pom < 0
            local_max_poz = np.diff(pom.astype(int)) > 0
            marker1[local_m[local_max_poz]] = True

    return marker1


def _detection_union(marker1, envelope, union_samples):

    marker1 = marker1.flatten('C')
    marker1 = np.asarray(marker1, dtype=bool)

    union_samples = np.ceil(union_samples)

    if math.fmod(union_samples, 2) == 0:
        union_samples = union_samples+1

    mask = np.ones((int(union_samples)))
    # dilatation
    marker1 = np.convolve(marker1, mask, mode='same') > 0
    # erosion
    marker1 = np.convolve(~marker1, mask, mode='same').astype(dtype=bool)
    marker1 = ~marker1

    marker2 = np.full(marker1.shape, False)
    non_zeros_1 = np.argwhere(np.diff(np.concatenate(([0], marker1))) > 0)
    non_zeros_2 = np.argwhere(np.diff(np.concatenate((marker1, [0]))) < 0)+1
    if non_zeros_1.size and non_zeros_2.size:
        point = np.vstack((non_zeros_1.T, non_zeros_2.T)).T
    else:
        point = np.array([])

    if point.size:
        for i in range(0, point.shape[0]):
            pom = envelope[point[i, 0]:point[i, 1]]
            maxp = [i for i, j in enumerate(pom) if j == max(pom)][0]
            marker2[point[i, 0]+maxp] = True

    return marker2


def _filt50Hz(d, fs, bandwidth, hum_fs=50):

    f0 = np.arange(hum_fs, fs/2, hum_fs)
    f0 = [x for x in f0 if x <= 1.1*bandwidth[1]]
    R = 1
    r = 0.985

    for i in range(0, len(f0)):
        b = np.array([1, -2*R*np.cos(2*np.pi*f0[i]/fs), R*R])
        a = np.array([1, -2*r*np.cos(2*np.pi*f0[i]/fs), r*r])

        d = filtfilt(b, a, d, axis=0, padtype='odd',
                     padlen=3*(max(len(b), len(a))-1))

    return d


class JancaDetector(Method):

    algorithm = 'JANCA_DETECTOR'
    algorithm_type = 'event'
    version = '1.0.0'

    dtype = [('event_peak', 'int32'),  # spike position
             # ('dur', 'float32'),  # spike duration - fix value 5 ms
             # ('condition', 'float32'),  # spike condition
             # ('weight', 'float32'),  # spike weight "CDF"
             # ('pdf', 'float32'),  # spike probability "PDF"
             # ('MV', 'float32'),  # type 1-obvious,0.5- ambiguous
             ('max_amplitude_env', 'float32'),  # max. amplitude of envelope
             # ('event_start', 'float32'),  # event start position
             # ('event_duration', 'float32'),  # event duration
             # ('MW', 'float32'),  # statistical weight
             # ('MPDF', 'float32'),  # statistical weight
             ('max_amplitude', 'float32')]

    def __init__(self, **kwargs):
        """
        Simplified python version of janca's EEG spike detector.
        {Janca et al. 2015} Reprogrammed from Matlab version by Daniel Uher.
        Modified by Jan Cimbalnik.

        Parameters
        ----------
        sig: np.ndarray
            1D numpy array of EEG data - NOT TRUE!!! check!!
        fs: int
            sampling frequency of the signal
        bandwidth: list
            low and high filtering boundaries (Hz)
        line_freq: int
            power line frequency to be filtered out (Hz). Default=50
        k1: float
            threshold value for obvious spikes. Default=3.65
        k2: float
            defines ambiguous spike threshold. Ambiguous is accepted when
            simultaneous obvious detection is in other channel k1>=k2.
            Default=k1
        k3: float
            decrease the threshold value k1*(mode+median)-k3(mean-mode).
            Default=0
        decimation: int
            decimation by number of samples for decimated signal
        polyspike_union_time: float
            Time to unify spikes into one detection on one channel.
            Default=0.12
        discharge_tol: float
            Time between spikes. Default=0.005
        """

        super().__init__(detect_spikes_janca, **kwargs)
