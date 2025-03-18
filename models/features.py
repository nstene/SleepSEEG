import numpy as np
import pandas as pd
import typing as t


class Features(np.ndarray):
    """A subclass of `numpy.ndarray` for handling and processing feature data.

    This class extends `numpy.ndarray` to provide additional functionality for
    feature processing, such as outlier removal, smoothing, and aggregation of
    nightly features.

    Attributes:
        Inherits all attributes from `numpy.ndarray`.

    Methods:
        remove_outliers(): Removes outliers from the feature data.
        smooth_features(): Applies smoothing to the feature data.
        get_nightly_features(): Aggregates features over a nightly period.
    """
    def __new__(cls, input_array: np.ndarray[float], *args, **kwargs):
        """Creates a new instance of the Features class.

        Args:
            input_array (array-like): Input data to be converted into a Features object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Features: A new instance of the Features class.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        """Finalizes the array creation process.

        This method is called when creating a view of the object. It ensures that
        any necessary attributes are copied over.

        Args:
            obj (numpy.ndarray): The original array being viewed.

        Returns:
            None
        """
        if obj is None:
            return
        pass

    def remove_outliers(self):
        """Removes outliers from the feature data using the IQR method.

        Outliers are identified using the interquartile range (IQR) and replaced
        with `NaN` values. A rolling window is used to calculate the moving average
        for outlier detection.

        :return: The modified array with outliers removed.
        """
        window_size = 10
        nf = self.shape[0]
        nch = self.shape[1]
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = self[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat = np.delete(arr=feat, obj=idx_nan)
                mov_avg = pd.Series(feat).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
                feat = feat - mov_avg
                q3 = np.percentile(feat, 75, method='closest_observation', axis=0)
                q1 = np.percentile(feat, 25, method='closest_observation', axis=0)
                iqr = q3 - q1
                lower_bound = q1 - 2.5 * iqr
                upper_bound = q3 + 2.5 * iqr
                outliers = (feat < lower_bound) | (feat > upper_bound) | idx_nan
                self[feat_idx, chan_idx, outliers] = np.nan
        return self


    def smooth_features(self):
        """Applies smoothing to the feature data using a rolling window.

        A rolling window is used to calculate the moving average, and the features
        are normalized by subtracting the mean and dividing by the standard deviation.

        :return: The modified array with smoothed features.
        """
        window_size = 3
        nf = self.shape[0]
        nch = self.shape[1]
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = self[feat_idx, chan_idx, :].copy()
                feat_copy = self[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat_copy = feat_copy[~idx_nan]
                filtered_feature = pd.Series(feat_copy).rolling(window=window_size, min_periods=1,
                                                                center=True).mean().to_numpy()
                feat[~idx_nan] = filtered_feature
                self[feat_idx, chan_idx, :] = (feat - np.nanmean(feat)) / (np.nanstd(feat, ddof=1))  # ddof =1 to match matlab's sample standard deviation (numpy's default is ddof = 0 for population standard deviation)
        return self


    def get_nightly_features(self):
        """Aggregates features over the whole recording using a rolling window.

        A rolling window is used to calculate the moving average of the features,
        and the normalized nightly features are computed.

        :return: An array of shape (nf, nch) containing the aggregated nightly features.
        """
        window_size = 10
        nf = self.shape[0]
        nch = self.shape[1]
        nightly_features = np.zeros(shape=(nf, nch))
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = self[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat = np.delete(arr=feat, obj=idx_nan)
                feat_avg = pd.Series(feat).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
                norm = np.linalg.norm(feat)
                if norm and not np.any(np.isnan(norm)):
                    nightly_features[feat_idx, chan_idx] = np.linalg.norm(feat_avg) / norm
                else:
                    print(feat_idx, chan_idx)

        return nightly_features