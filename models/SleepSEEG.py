import warnings
import logging
import typing as t
import os
import cProfile

from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import signal

from eeg_reader import EdfReader, EEG, Montage, Channel
from models.MatlabModelImport import MatlabModelImport, ClassificationTree

class Epoch(EEG):
    def __init__(self, data, timestamps, fs, montage: Montage = None):
        super().__init__(data=data, timestamps=timestamps, montage=montage)
        self.timestamps = timestamps
        self.fs = fs
        self.features = None

    def trim(self, n_samples_from_start: int, n_samples_to_end: int):
        if len(self._data.shape) > 1:
            trimmed_data = self._data[:, n_samples_from_start:-n_samples_to_end]
        else:
            trimmed_data = self._data[n_samples_from_start:-n_samples_to_end]
        self._data = trimmed_data
        return

    @property
    def data(self):
        return self._data

    def change_sampling_rate(self):
        """The resulting sample rate is up / down times the original sample rate.

        """
        target_fs = 256
        up = target_fs
        down = self.fs
        if len(self._data.shape) > 1:
            axis = 1
        else:
            axis = 0
        self._data = signal.resample_poly(self._data, up, down, axis=axis)
        self.fs = target_fs
        return

    def mean_normalize(self):
        normalized_data = self._data - np.mean(self._data, axis=0)
        self._data = normalized_data
        return

    def compute_features(self) -> np.ndarray:
        features_list = []
        for i in range(len(self._montage.channels)):
            x = self._data[i]
            features_list.append(self._compute_signal_features(data=x))
        self.features = np.vstack(features_list)
        return self.features

    @staticmethod
    def _compute_signal_features(data: np.ndarray):
        """
                Computes the features for the given signal chunk on a given channel.
                We use a bandpass filter, iteratively for each scale from 1 to J.
                The input signal is truncated to a length that is a multiple of 2^J to ensure it can be properly decomposed into
                the specific number of scales.

                Wavelet decomposition:
                    - High pass filtering
                        The high-pass filter is applied to the signal to extract the detail coefficients d, which represent the
                    high-frequency components of the signal.
                    - Low pass filtering
                        The low-pass filter is applied to the signal to extract the approximation coefficients x, which represent
                        the low-frequency components of the signal
                    - Downsampling:
                        After each filtering step, the signal is downsampled by a factor 2, effectively halving the sampling rate
                        for the next iteration.

                Wavelet Leaders:
                    - Wavelet leaders calculation:
                        The function calculates the wavelets leaders, which are the maximum absolute values of the wavelet
                        coefficients within a certain neighbourhood. This is done to capture the local maxima of the signal's
                        wavelet coefficients.
                    - Transient removal:
                        The function discards the initial and final parts of the wavelet leaders to avoid edge effects (transients).

                Feature extraction:
                    - Logarithmic moments:
                        The function computes the logarithmic moments of the wavelet leaders. Specifically, it calculates the
                        mean u1, variance u2 - u1^2, and skewness u3 - 3u1u2 + 2u1^3 of the logarithm of the wavelet leaders.
                        https://math.stackexchange.com/questions/2048072/what-is-the-meaning-of-log-moment
                    - Cumulants:
                        These moments are stored in a matrix C, which will be used later to compute the final features.
                    - Weighted fit:
                        The function computes weigths w for a weighted linear fit of the cumulants across scales.
                        This is done to emphasize certain scales more than others based on their importance.

                Final feature vector
                    - Feature vector construction:
                        The final feature vector f is constructed by combining the weighted cumulants and the wavelet
                        coefficients. The feature vector includes:
                            - The weighted cumulants C*w, which are scaled by log2(exp(1)) to convert from natural logarithm to
                              base-2 logarithm
                            - The wavelet coefficients mwc, which are normalized and transformed using a logarithm scale.
                    - Exclusion of certain scales:
                        The function excludes features corresponding to highest frequency scales (64-128Hz) from the final
                        feature vector.

                Summary
                The compute_features function extracts a set of features from an iEEG signal using wavelet decomposition and
                wavelet leaders. These features capture the signal's frequency content and statistical properties across
                different scales, which are then used for sleep stage classification. The function is designed to be robust to
                transient effects and emphasizes certain frequency bands that are likely important for distinguishing between
                different sleep stages.

                Use numpy and pywt

                :return:
                """
        # Compute for one epoch, then loop over all epochs
        # epoch_duration = 30
        # total_epoch_duration_samples = ((epoch_duration + self._time_window) * self._edf.sampling_frequency
        #                         - self._initial_buffer.shape[1])

        # Scales for wavelet decomposition. Signal is decomposed into 8 different frequency band.
        # widest is 0.5-1Hz (since sampling rate is 256)
        J = 8
        nd = 5  # Filter order
        scales = np.arange(1, 6)  # Selected scales (2 to 6)

        low_pass_filter_coeff = np.array(
            [.22641898, .85394354, 1.02432694, .19576696, -.34265671, -.04560113, .10970265,
             -.0088268, -.01779187, .00471742793])
        high_pass_filter_coeff = low_pass_filter_coeff[::-1] * (-1) ** np.arange(2 * nd)
        # TODO: Why these values for high-pass?

        # Ensure signal length is multiple of 2^J. // is the floor division
        data = data[:2 ** J * (len(data) // 2 ** J)]
        flattened_data = data.flatten()

        # Initialize arrays before looping over the scales
        laa = np.zeros(int(len(data)/2))
        C = np.zeros((3, J))  # Cumulants
        b = np.ones(J)  # Weights for scales
        mwc = np.zeros(3 * J)  # Wavelet coefficients

        # Wavelet decomposition and feature extraction
        # Use pywt for wavelet transforms: https://pywavelets.readthedocs.io/en/latest/
        for scale in range(J):
            # High-pass filetering (detail coefficients)
            d = signal.lfilter(high_pass_filter_coeff, 1, data)
            lea = d[::2]

            data = signal.lfilter(low_pass_filter_coeff, 1, data)
            data = data[::2]

            start = int(nd + np.ceil(256 / 2 ** (scale + 1))) - 1
            end = int(np.maximum(np.ceil(256 / 2 ** (scale + 1)) - nd, 0))
            if end == 0:
                mm = np.abs(lea[start:])
            else:
                mm = np.abs(lea[start: -end])

            # Wavelet coefficients
            mwc[scale] = np.log(np.mean(mm))
            mwc[scale + J] = np.sum(mm ** 2)
            mwc[scale + 2 * J] = np.log(np.std(mm))

            lea = np.insert(lea, 0, 0)
            lea = np.append(lea, 0)
            lea = np.abs(lea)
            lea = np.maximum(np.maximum(lea[:-2], lea[1:-1]), lea[2:])
            lea = np.maximum(lea, laa)
            laa = np.maximum(lea[::2], lea[1::2])

            if end == 0:
                lea = lea[start:]
            else:
                lea = lea[start: -end]

            # Log-transform and cumulants
            le = np.log(lea)
            u1 = np.mean(le)
            u2 = np.mean(le ** 2)
            u3 = np.mean(le ** 3)
            C[:, scale] = [u1, u2 - u1 ** 2, u3 - 3 * u1 * u2 + 2 * u1 ** 3]

            # Update weights
            nj = len(lea)
            b[scale] = nj

        C = C[:, scales]
        b = b[scales]
        # Weighted fit for cumulants
        V0 = np.sum(b[scales - 1])
        V1 = np.dot(scales + 1, b[scales - 1])
        V2 = np.dot((scales + 1) ** 2, b[scales - 1])
        w = b[scales - 1] * ((V0 * (scales + 1) - V1) / (V0 * V2 - V1 ** 2))

        # Compute final features
        f = np.hstack([
            np.log2(np.exp(1)) * np.dot(C, w),  # Weighted cumulants
            mwc
        ])

        # Exclude the 64-128 Hz scale (4th scale)
        exclude_indices = [3, 3 + J, 3 + 2 * J]
        f = np.delete(f, exclude_indices)

        # Normalize wavelet coefficients
        J = J - 1  # Adjust J after exclusion
        f[J + 3:2 * J + 3] = np.log10(f[J + 3:2 * J + 3] / np.sum(f[J + 3:2 * J + 3]))

        return f

class SleepSEEG:
    def __init__(self, filepath: str):
        self.features = None
        logger = logging.getLogger('SleepSEEG_logger')
        logger.setLevel(logging.DEBUG)  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        # Create console handler and set its format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Log all messages from DEBUG level and up
        # Define log format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        # Add handler to the logger
        logger.addHandler(console_handler)

        self.logger = logger
        self._initial_buffer = None
        self._time_window = None
        self._filepath = filepath
        self._edf = EdfReader(filepath=filepath)
        self.montage_type = 'referential'
        self.epochs = None
        self.epochs_preprocessed = False
        self.sleep_stage = None

    def read_initial_buffer(self, time_window: float = 2.5):
        window_center = self._edf.start_time_sample
        window_length_samples = time_window * self._edf.sampling_frequency
        start_sample = window_center - round(window_length_samples / 2)
        stop_sample = window_center + round(window_length_samples / 2)
        self._time_window = time_window
        self._initial_buffer = self._edf.extract_data_pyedflib(start_sample=start_sample, stop_sample=stop_sample)
        return

    def get_epoch_by_index(self, epoch_index: int):
        rolling_window_seconds = 2.5
        epoch_zero_start = int(self._edf.start_time_sample - rolling_window_seconds / 2 * self._edf.sampling_frequency)

        epoch_start = epoch_zero_start + epoch_index * 30 * self._edf.sampling_frequency
        epoch_end = (epoch_zero_start + (epoch_index + 1) * 30 * self._edf.sampling_frequency
                     + rolling_window_seconds * self._edf.sampling_frequency)

        return self.read_epoch(start_sample=epoch_start, end_sample=epoch_end)

    def compute_epoch_features(self):
        features_list = []
        for epoch_index in tqdm(range(self._edf.n_epochs), "Computing features for epoch"):
            # Read epoch
            epoch = self.get_epoch_by_index(epoch_index=epoch_index)

            # Make bipolar montage
            epoch.make_bipolar_montage()

            # Preprocess epoch
            epoch.change_sampling_rate()
            epoch.trim(n_samples_from_start=int(0.25*epoch.fs), n_samples_to_end=int(0.25*epoch.fs))
            epoch.mean_normalize()

            # Compute epoch features
            features_list.append(epoch.compute_features())

        return np.array(features_list).transpose(2, 1, 0)

    def read_epoch(self, start_sample: int, end_sample: int = None, epoch_duration: float = None):
        if not end_sample:
            end_sample = int(start_sample + epoch_duration * self._edf.sampling_frequency)

        if (end_sample % 1 != 0) or (start_sample % 1 != 0):
            raise ValueError(f"Start and end samples must be integers. "
                             f"start_sample:{start_sample}; end_sample: {end_sample}")

        start_sample = int(start_sample)
        end_sample = int(end_sample)

        data, timestamps, channels = self._edf.extract_data_mne(start_sample=start_sample, end_sample=end_sample)
        #data, channels = self._edf.extract_data_pyedflib(start_sample=start_sample, stop_sample=end_sample)
        montage = Montage(channels=channels, montage_type='referential')

        return Epoch(data=data, fs=self._edf.sampling_frequency, timestamps=timestamps, montage=montage)

    @staticmethod
    def remove_outliers(features: np.ndarray):
        """Remove outliers.
        """
        nf = features.shape[0]
        nch = features.shape[1]
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = features[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat = np.delete(arr=feat, obj=idx_nan)
                mov_avg = uniform_filter1d(feat, size=min(feat.shape[0], 10), axis=0, mode="wrap")
                feat = feat - mov_avg
                q3 = np.percentile(feat, 75, method='lower', axis=0)
                q1 = np.percentile(feat, 25, method='lower', axis=0)
                iqr = q3 - q1
                lower_bound = q1 - 2.5 * iqr
                upper_bound = q3 + 2.5 * iqr
                outliers = (feat < lower_bound) | (feat > upper_bound) | idx_nan
                features[feat_idx, chan_idx, outliers] = np.nan
        return features

    @staticmethod
    def smooth_features(features: np.ndarray):
        """Smooth features.
        """
        window_size = 3
        nf = features.shape[0]
        nch = features.shape[1]
        ne = features.shape[2]
        features_smoothed_normalized = np.ndarray(shape=features.shape)
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = features[feat_idx, chan_idx, :].copy()
                feat_copy = features[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat_copy = feat_copy[~idx_nan]
                filtered_feature = pd.Series(feat_copy).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
                feat[~idx_nan] = filtered_feature
                features_smoothed_normalized[feat_idx, chan_idx, :] = (feat - np.nanmean(feat)) / (np.nanstd(feat))
        return features_smoothed_normalized

    @staticmethod
    def get_nightly_features(features: np.ndarray):
        window_size = 10
        nf = features.shape[0]
        nch = features.shape[1]
        ne = features.shape[2]
        nightly_features = np.zeros(shape=(nf, nch))
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = features[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat = np.delete(arr=feat, obj=idx_nan)
                feat_avg = pd.Series(feat).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
                norm = np.linalg.norm(feat)
                if norm and not np.any(np.isnan(norm)):
                    if norm == 0:
                        print('oh')
                    if np.count_nonzero(np.isnan(feat_avg))>0:
                        print('hey')
                    nightly_features[feat_idx, chan_idx] = np.linalg.norm(feat_avg) / norm
                else:
                    print(feat_idx, chan_idx)

        return nightly_features

    def preprocess_features(self, features: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features. Features should be an array of shape (Nfeat, Nchans, Nepochs)
        :param features:
        :return:
        """

        # Outlier detection: remove outliers comparing the same feature on same channel comparing across epoch axis
        features_without_outliers = self.remove_outliers(features=features)

        # Smoothing & Normalizing
        preprocessed_features = self.smooth_features(features=features_without_outliers)

        # Extract nightly features
        nightly_features = self.get_nightly_features(features=preprocessed_features)

        return preprocessed_features, nightly_features

    def cluster_channels(self, nightly_features: np.ndarray, gc):
        Nch = nightly_features.shape[1]
        channel_groups = np.zeros(Nch, dtype=int)
        Ng = gc.shape[0]

        for nc in range(Nch):
            differences = np.tile(nightly_features[:, nc], (Ng, 1)) - gc
            distances = np.sum(differences ** 2, axis=1)  # Compute squared Euclidean distances
            channel_groups[nc] = np.argmin(distances)  # Find index of minimum distance

        return channel_groups

    def score_epochs(self, features, models: t.List[t.List[ClassificationTree]], channel_groups):
        """
        For each cluster group, find channels that belong to that group
        Extract their features, reshape them, remove rows that have only NaNs
        Apply the four machine learning models to predict probabilities prob
        Compute average posterior probabilities: output is confidence, the normalized probabilities

        :param features: (N_features x N_channels x N_epochs)
        :param models:
        :param channel_groups:
        :return:
        """

        def _compute_confidence(prob):
            """
            Computes the final confidence matrix by averaging across channels and normalizing.
            """
            prop = np.nanmean(prob, axis=0)  # Average over channels
            prop[:, :4] /= np.sum(prop[:, :4], axis=1, keepdims=True)  # Normalize first 4 columns
            confidence = prop
            return confidence

        def _score_channels(models, features, logger):
            n_features, n_channels, n_epochs = features.shape
            n_stages = 7
            feat = np.transpose(features, (1, 2, 0))
            postprob = np.zeros(shape=(n_channels, n_epochs, n_stages))
            clusters = np.unique(channel_groups).astype(int)
            logger.info(msg=f"{len(clusters)} clusters found.")
            for cluster_idx in tqdm(clusters, "Computing probabilities for cluster"):
                ik = channel_groups == cluster_idx
                if np.any(ik):
                    fe = feat[ik, :, :].reshape(np.count_nonzero(ik) * n_epochs, n_features)
                    del_rows = np.all(np.isnan(fe), axis=1)  # Rows with all NaNs

                    prob = np.full((fe.shape[0], 7), np.nan)  # Initialize probabilities

                    # Predict using each model
                    prob[~del_rows, :4] = models[cluster_idx][0].predict_proba_deepseek(fe[~del_rows])
                    prob[~del_rows, 4] = models[cluster_idx][1].predict_proba_deepseek(fe[~del_rows])[:, 0]
                    prob[~del_rows, 5] = models[cluster_idx][2].predict_proba_deepseek(fe[~del_rows])[:, 0]
                    prob[~del_rows, 6] = models[cluster_idx][3].predict_proba_deepseek(fe[~del_rows])[:, 0]

                    # Reshape and assign to postprob
                    postprob[ik] = prob.reshape(-1, n_epochs, 7)
                return postprob

        def _define_stage(confidence):
            # Define stage for each epoch
            mm = np.max(confidence[:, :4], axis=1)  # maximum confidence value
            sa = np.argmax(confidence[:, :4], axis=1) + 1  # stage index

            # handle nans
            sa[np.isnan(mm)] = 2
            mm[np.isnan(mm)] = 0

            sa[sa > 2] += 1

            # adjust stage based on confidence thresholds
            sa[(sa == 1) & (confidence[:, 4] < 0.5)] = 3
            sa[(sa == 2) & (confidence[:, 5] < 0.5)] = 3
            sa[(sa == 4) & (confidence[:, 6] < 0.5)] = 3

            return sa, mm

        postprob = _score_channels(models=models, features=features, logger=self.logger)
        confidence = _compute_confidence(prob=postprob)
        sa, mm = _define_stage(confidence=confidence)
        return sa, mm


class SleepStage:
    def __init__(self, stage, max_confidence, sample, time):
        self.stage=stage
        self.max_confidence = max_confidence
        self.sample = sample
        self.time = time





if __name__ == "__main__":
    filepath = r'../eeg_data/auditory_stimulation_P18_002_3min.edf'
    sleep_eeg_instance = SleepSEEG(filepath=filepath)

    cProfile.run('sleep_eeg_instance.compute_epoch_features()')
    features = sleep_eeg_instance.compute_epoch_features()
    features_preprocessed, nightly_features = sleep_eeg_instance.preprocess_features(features=features)

    np.save(file='features_processed_full.npy', arr=features_preprocessed, allow_pickle=False)
    loaded_processed = np.load(file='features_processed_full.npy')
    np.array_equal(features_preprocessed, loaded_processed)

    parameters_directory = r'../model_parameters'
    model_filename = r'Model_BEA_refactored.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)

    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)
    matlab_model_import = MatlabModelImport(model_filepath=model_filepath, gc_filepath=gc_filepath)

    channel_groups = sleep_eeg_instance.cluster_channels(nightly_features=features_preprocessed, gc=matlab_model_import.GC)
    print('hi')