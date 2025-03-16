import typing as t
from scipy.signal import firwin, lfilter
from scipy import signal
import numpy as np
from numba import njit

from models.layout import EEG, Montage, Channel


class Epoch(EEG):
    """A class representing an epoch of EEG data.

    An epoch is a segment of EEG data with a fixed duration. This class provides functionality
    for preprocessing EEG epochs and extracting features.

    Attributes:
        fs (float): Sampling frequency of the EEG data.
        features (np.ndarray): Computed features for the epoch.
        start_time (float): Start time of the epoch in seconds.
        start_sample (int): Start sample index of the epoch in the original data.
        _data (np.ndarray): The EEG data for the epoch (channels x samples).
        montage (Montage): The montage describing the channel layout.
        timestamps (np.ndarray): Timestamps for each sample in the epoch.

    Methods:
        trim(n_samples_from_start, n_samples_to_end): Trims the epoch by removing samples
            from the start and end.
        change_sampling_rate(): Resamples the epoch to a target sampling rate (256 Hz).
        change_sampling_rate_2(): Resamples the epoch using predefined resampling factors.
        change_sampling_rate_deepseek(): Resamples the epoch using custom FIR filters.
        mean_normalize(): Normalizes the epoch data by subtracting the mean.
        compute_features(): Computes features for the epoch using wavelet decomposition.
        process_epoch(): Processes the epoch by applying montage, resampling, trimming,
            and normalization.
        drop_channels(channels_to_exclude): Removes specified channels from the epoch.
    """

    def __init__(self, data, fs, timestamps=None, montage: Montage = None):
        """Initializes the Epoch instance.

        :param data: The EEG data for the epoch (channels x samples).
        :param fs: Sampling frequency of the EEG data.
        :param timestamps: Timestamps for each sample in the epoch.
        :param montage: The montage describing the channel layout.
        """
        super().__init__(data=data, timestamps=timestamps, montage=montage)
        self.fs = fs
        self.features = None
        self.start_time = None
        self.start_sample = None

    def trim(self, n_samples_from_start: int, n_samples_to_end: int):
        """Trims the epoch by removing samples from the start and end.

        :param n_samples_from_start: Number of samples to remove from the start.
        :param n_samples_to_end: Number of samples to remove from the end.
        """
        if len(self._data.shape) > 1:
            trimmed_data = self._data[:, n_samples_from_start:-n_samples_to_end]
        else:
            trimmed_data = self._data[n_samples_from_start:-n_samples_to_end]
        self._data = trimmed_data
        return

    @property
    def data(self):
        """Returns the EEG data for the epoch.

        :return: The EEG data (channels x samples)
        """
        return self._data

    def set_data(self, data):
        """Sets the EEG data for the epoch.

        :param data: The EEG data (channels x epochs)
        """
        self._data = data
        return

    def change_sampling_rate(self):
        """Resamples the epoch to a target sampling rate of 256 Hz.
        This is the exact same implementation as the MATLAB version.
        """
        # TODO: maker sure method works if data is 1-dimensional
        # Define the filter coefficients
        od5 = 54
        od2 = 26
        ou2 = 34

        hd5 = np.array([
            -0.000413312132792, 0.000384910656353, 0.000895384486596, 0.001426584098180, 0.001572675788393,
            0.000956099017099, -0.000559378457343, -0.002678217568221, -0.004629975982837, -0.005358589238386,
            -0.003933117464092, -0.000059710059922, 0.005521319363883, 0.010983495478404, 0.013840996082966,
            0.011817315106321, 0.003905283425021, -0.008768844009700, -0.022682212400564, -0.032498023687148,
            -0.032456772047175, -0.018225658085891, 0.011386634156651, 0.053456542440034, 0.101168250947271,
            0.145263694388270, 0.176384224234024, 0.187607302744229, 0.176384224234024, 0.145263694388270,
            0.101168250947271, 0.053456542440034, 0.011386634156651, -0.018225658085891, -0.032456772047175,
            -0.032498023687148, -0.022682212400564, -0.008768844009700, 0.003905283425021, 0.011817315106321,
            0.013840996082966, 0.010983495478404, 0.005521319363883, -0.000059710059922, -0.003933117464092,
            -0.005358589238386, -0.004629975982837, -0.002678217568221, -0.000559378457343, 0.000956099017099,
            0.001572675788393, 0.001426584098180, 0.000895384486596, 0.000384910656353, -0.000413312132792
        ])

        hd2 = np.array([
            0.001819877350937, 0.000000000000000, -0.005222562671417, -0.000000000000000, 0.012064004143824,
            0.000000000000000, -0.024375517671448, -0.000000000000001, 0.046728257943321, 0.000000000000001,
            -0.095109546093322, -0.000000000000001, 0.314496714630999, 0.500000000000001, 0.314496714630999,
            -0.000000000000001, -0.095109546093322, 0.000000000000001, 0.046728257943321, -0.000000000000001,
            -0.024375517671448, 0.000000000000000, 0.012064004143824, -0.000000000000000, -0.005222562671417,
            0.000000000000000, 0.001819877350937
        ])

        hu2 = np.array([
            -0.000436344318144, 0.000871220095338, 0.002323148230146, 0.002161775863027, -0.001491947315250,
            -0.006921578214988, -0.008225251275241, -0.000175549247298, 0.014346107990165, 0.022307211309135,
            0.009627222866552, -0.023067228777693, -0.052048618204333, -0.041386934088038, 0.030232588120746,
            0.146540456844067, 0.255673093905358, 0.300317753050023, 0.255673093905358, 0.146540456844067,
            0.030232588120746, -0.041386934088038, -0.052048618204333, -0.023067228777693, 0.009627222866552,
            0.022307211309135, 0.014346107990165, -0.000175549247298, -0.008225251275241, -0.006921578214988,
            -0.001491947315250, 0.002161775863027, 0.002323148230146, 0.000871220095338, -0.000436344318144
        ])

        fs = round(self.fs)

        X = self.data.T

        if fs == 200:
            z = np.zeros((ou2 // 2, X.shape[1]))
            X = 2 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X)], axis=1), (2 * X.shape[0], X.shape[1])))
            X = lfilter(hu2, 1, np.vstack([X, z]))  # Filter
            X = X[ou2 // 2:, :]  # Remove delay
            z = np.zeros((od5 // 2, X.shape[1]))
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5

        elif fs == 256:
            # No change necessary
            pass

        elif fs == 500:
            z = np.zeros((od5 // 2, X.shape[1]))
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5

        elif fs == 512:
            z = np.zeros((od2 // 2, X.shape[1]))
            X = lfilter(hd2, 1, np.vstack([X, z]))  # Filter
            X = X[od2 // 2:, :]  # Remove delay
            X = X[::2, :]  # Downsample x2

        elif fs == 1000:
            z = np.zeros((od5 // 2, X.shape[1]))
            X = 2 * np.reshape(
                np.transpose(np.stack([X, np.zeros_like(X)], axis=1), (2 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5

        elif fs == 1024:
            z = np.zeros((od2 // 2, X.shape[1]))
            X = lfilter(hd2, 1, np.vstack([X, z]))  # Filter
            X = X[od2 // 2:, :]  # Remove delay
            X = X[::2, :]  # Downsample x2
            X = lfilter(hd2, 1, np.vstack([X, z]))  # Filter
            X = X[od2 // 2:, :]  # Remove delay
            X = X[::2, :]  # Downsample x2

        elif fs == 2000:
            z = np.zeros((od5 // 2, X.shape[1]))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5
            X = 4 * np.reshape(np.transpose(np.stack([X, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=1),
                                            (4 * X.shape[0], X.shape[1])))
            X = lfilter(hd5, 1, np.vstack([X, z]))  # Filter
            X = X[od5 // 2:, :]  # Remove delay
            X = X[::5, :]  # Downsample x5

        elif fs == 2048:
            od2 = len(hd2)  # Filter order
            z = np.zeros((od2 // 2, X.shape[1]))  # Zero padding

            # Apply the filter and downsample iteratively
            X = lfilter(hd2, 1, np.vstack([X, z]), axis=0)  # Filter
            X = X[od2 // 2:]  # Remove initial transient
            X = X[::2]  # Downsample by 2

            X = lfilter(hd2, 1, np.vstack([X, z]), axis=0)
            X = X[od2 // 2:]
            X = X[::2]

            X = lfilter(hd2, 1, np.vstack([X, z]), axis=0)
            X = X[od2 // 2:]
            X = X[::2]

        else:
            raise ValueError(f"Unsupported sampling rate: {fs}")
        self._data = X.T
        self.fs = 256
        return

    def mean_normalize(self):
        """Normalizes the epoch data by substracting each signal's mean."""
        if len(self.data.shape) < 2:
            normalized_data = self._data - np.mean(self._data)
        else:
            normalized_data = self._data - np.tile(np.mean(self._data, axis=1), (self._data.shape[1], 1)).T
        self._data = normalized_data
        return

    def compute_features(self) -> np.ndarray:
        """Computes the features of each signal."""
        features_list = []
        for i in range(len(self.montage.channels)):
            features_list.append(self._compute_signal_features(data=self.data[i]))
        self.features = np.vstack(features_list)
        return self.features

    @staticmethod
    def _compute_signal_features(data: np.ndarray):
        """
        Computes the features for a given epoch on a given channel.

        :param data: EEG single channel signal (1 x samples)
        :return: features for that channel (1 x 24)
        """

        # Scales for wavelet decomposition. Signal is decomposed into 8 different frequency band.
        # widest is 0.5-1Hz (since sampling rate is 256)
        J = 8
        nd = 5  # Filter order
        scales = np.arange(1, 6)  # Selected scales (2 to 6)

        low_pass_filter_coeff = np.array(
            [.22641898, .85394354, 1.02432694, .19576696, -.34265671, -.04560113, .10970265,
             -.0088268, -.01779187, .00471742793])
        high_pass_filter_coeff = low_pass_filter_coeff[::-1] * (-1) ** np.arange(2 * nd)
        # TODO: Why these values?

        # Ensure signal length is multiple of 2^J
        data = data[:2 ** J * (len(data) // 2 ** J)]

        # Initialize arrays before looping over the scales
        laa = np.zeros(int(len(data)/2))
        C = np.zeros((3, J))  # Cumulants
        b = np.ones(J)  # Weights for scales
        mwc = np.zeros(3 * J)  # Wavelet coefficients

        # Wavelet decomposition and feature extraction
        # Use pywt for wavelet transforms: https://pywavelets.readthedocs.io/en/latest/
        for scale in range(J):
            # High-pass filtering (detail coefficients)
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
            np.log2(np.exp(1)) * np.dot(C, w),
            mwc
        ])

        # Exclude the 64-128 Hz scale (4th scale)
        exclude_indices = [3, 3 + J, 3 + 2 * J]
        f = np.delete(f, exclude_indices)

        # Normalize wavelet coefficients
        J = J - 1  # Adjust J after exclusion
        f[J + 3:2 * J + 3] = np.log10(f[J + 3:2 * J + 3] / np.sum(f[J + 3:2 * J + 3]))

        return f

    def process_epoch(self):
        """Preprocess an epoch.
        The epoch data is first re-calculated to make it bipolar if the montage was originally referential.
        The sampling rate is then changed to a standard 256 Hz.
        The epoch is trimmed such that it is exactly 30s long.
        The epoch data is then normalized.
        """
        self.make_bipolar_montage()
        self.change_sampling_rate()
        self.trim(n_samples_from_start=int(0.25 * self.fs), n_samples_to_end=int(0.25 * self.fs))
        self.mean_normalize()
        return

    def drop_channels(self, channels_to_exclude: t.List[Channel]):
        """Excludes from the data the specified channels."""
        excluded_channel_names = [chan_to_exclude.name for chan_to_exclude in channels_to_exclude]
        # Drop channels
        idx_to_drop = [self.chan_idx[name] for name in excluded_channel_names]
        self.set_data(np.delete(self._data, idx_to_drop, axis=0))

        # Update montage
        self.montage.channels = np.delete(self.montage.channels, idx_to_drop, axis=0).tolist()

        # Update channel indices
        updated_dict = {name: idx for name, idx in self.chan_idx.items() if name not in excluded_channel_names}
        sorted_channels = sorted(updated_dict.items(), key=lambda x: x[1])
        updated_dict = {name: new_idx for new_idx, (name, _) in enumerate(sorted_channels)}
        self.set_chan_idx(updated_dict)


