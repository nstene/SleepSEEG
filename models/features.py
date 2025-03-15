import numpy as np
import pandas as pd
import typing as t


class Features(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        # Convert input to ndarray
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        # This is called when creating a view of the object
        if obj is None:
            return
        # Copy any attributes if needed
        pass

    def remove_outliers(self):
        """Remove outliers.
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
        """Smooth features.
        """
        window_size = 3
        nf = self.shape[0]
        nch = self.shape[1]
        ne = self.shape[2]
        for feat_idx in range(nf):
            for chan_idx in range(nch):
                feat = self[feat_idx, chan_idx, :].copy()
                feat_copy = self[feat_idx, chan_idx, :].copy()
                idx_nan = np.isnan(feat)
                feat_copy = feat_copy[~idx_nan]
                filtered_feature = pd.Series(feat_copy).rolling(window=window_size, min_periods=1,
                                                                center=True).mean().to_numpy()
                feat[~idx_nan] = filtered_feature
                self[feat_idx, chan_idx, :] = (feat - np.nanmean(feat)) / (np.nanstd(feat,
                                                                                                             ddof=1))  # ddof =1 to match matlab's sample standard deviation (numpy's default is ddof = 0 for population standard deviation)
        return self


    def get_nightly_features(self):
        window_size = 10
        nf = self.shape[0]
        nch = self.shape[1]
        ne = self.shape[2]
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