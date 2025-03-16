from models.layout import EEG, Montage
from scipy.ndimage import uniform_filter1d
from scipy.signal import firwin, lfilter
from scipy import signal
import numpy as np
from numba import njit


class Epoch(EEG):
    def __init__(self, data, fs, timestamps=None, montage: Montage = None):
        super().__init__(data=data, timestamps=timestamps, montage=montage)
        self.fs = fs
        self.features = None
        self.start_time = None
        self.start_sample = None

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

    def change_sampling_rate_2(self):
        resample_factors = {
            200: (8, 20),  # Equivalent to x2 up, x5 down, x4 up, x5 down
            256: (1, 1),  # No change
            500: (32, 125),  # Equivalent to x4 up, x5 down, x4 up, x5 down, x4 up, x5 down
            512: (1, 2),  # Downsample x2
            1000: (8, 20),  # Equivalent to x2 up, x5 down, x4 up, x5 down, x4 up, x5 down
            1024: (1, 4),  # Downsample x2 twice
            2000: (8, 40),  # Equivalent to x5 down, x4 up, x5 down, x4 up, x5 down
            2048: (1, 8)  # Downsample x2 three times
        }

        if round(self.fs) not in resample_factors:
            raise ValueError(
                f"Error: Sampling rate {self.fs} not supported. Supported rates: {list(resample_factors.keys())}")

        up, down = resample_factors[round(self.fs)]
        self._data = self.matlab_resample(self._data, up, down)
        self.fs = 256

    @staticmethod
    def matlab_resample(X, up, down):
        # Design a low-pass FIR filter similar to MATLAB
        GCD = np.gcd(up, down)
        up, down = up // GCD, down // GCD  # Reduce fraction

        # MATLAB uses a Kaiser window with β=5 (standard value in MATLAB's resample)
        num_taps = 10 * max(up, down)  # Filter length (MATLAB default)
        beta = 5  # Kaiser window beta

        h = firwin(num_taps, 1.0 / max(up, down), window=('kaiser', beta))

        # Upsample by inserting zeros
        X_up = np.zeros((len(X) * up, *X.shape[1:]))
        X_up[::up] = X  # Insert zeros in between

        # Apply the FIR filter
        X_filt = lfilter(h, 1.0, X_up, axis=0)

        # Downsample by selecting every `down`-th sample
        X_resampled = X_filt[::down]

        return X_resampled

    def change_sampling_rate_deepseek(self):
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
        if len(self.data.shape) < 2:
            normalized_data = self._data - np.mean(self._data)
        else:
            normalized_data = self._data - np.tile(np.mean(self._data, axis=1), (self._data.shape[1], 1)).T
        self._data = normalized_data
        return

    def compute_features(self) -> np.ndarray:
        features_list = []
        for i in range(len(self._montage.channels)):
            features_list.append(self._compute_signal_features(data=self.data[i]))
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