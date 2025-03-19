import unittest
import numpy as np
import h5py
import os

from models.sleep_seeg import SleepSEEG, Epoch
from models.features import Features
from models.matlab_adaptator import MatlabModelImport

def rmse(signal1, signal2):
    return np.sqrt(np.mean((signal1 - signal2) ** 2))

if os.getcwd().split('\\')[-1] == 'test':
    os.chdir("..")  # Go one directory up


class TestSleepSEEG(unittest.TestCase):
    epoch_0_file = r'matlab_files/epoch_0_LTP1.mat'
    epoch_0_resampled_file = r'matlab_files/epoch_0_LTP1_resampled.mat'
    epoch_0_resampled_trimmed_file = r'matlab_files/epoch_0_LTP1_resampled_trimmed.mat'
    epoch_0_resampled_trimmed_baselined_file = r'matlab_files/epoch_0_LTP1_resampled_trimmed_baselined.mat'
    full_data_file = r'eeg_data/auditory_stimulation_P18_002.edf'
    data_file_3min = r'eeg_data/auditory_stimulation_P18_002_3min.edf'
    features_unprocessed_all_file = r'matlab_files/features_unprocessed_v2_time.mat'
    features_processed_all_file = r'matlab_files/features_processed_v2_time.mat'
    nightly_features_processed_matlab_file = r'matlab_files/featfeat.mat'
    nightly_features_all_file = r'matlab_files/nightly_features_v2_time.mat'
    channel_groups_file = r'matlab_files/channel_groups.mat'
    sa_all_file = r'matlab_files/sa.mat'
    mm_all_file = r'matlab_files/mm.mat'
    postprob_all_file = r'matlab_files/postprob.mat'

    parameters_directory = r'model_parameters'
    model_filename = r'Model_BEA_full.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)

    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)

    @classmethod
    def setUpClass(cls) -> None:
        with h5py.File(cls.epoch_0_file, 'r') as file:
            cls.epoch_0_matlab_data = list(file['x_save'])[0]
        with h5py.File(cls.epoch_0_resampled_file, 'r') as file:
            cls.epoch_0_matlab_data_resampled = list(file['x_save'])[0]
        with h5py.File(cls.epoch_0_resampled_trimmed_file, 'r') as file:
            cls.epoch_0_matlab_data_resampled_trimmed = list(file['x_save'])[0]
        with h5py.File(cls.epoch_0_resampled_trimmed_baselined_file, 'r') as file:
            cls.epoch_0_matlab_data_resampled_trimmed_baselined = list(file['x_save'])[0]
        with h5py.File(cls.features_unprocessed_all_file, 'r') as file:
            features_unprocessed_matlab = list(file['feature'])
        features_unprocessed_matlab = np.array(features_unprocessed_matlab)
        cls.features_unprocessed_matlab = np.transpose(features_unprocessed_matlab, (0, 2, 1))
        with h5py.File(cls.nightly_features_processed_matlab_file, 'r') as file:
            nightly_features_processed_matlab = list(file['featfeat'])
        cls.nightly_features_processed_matlab = np.array(nightly_features_processed_matlab)
        with h5py.File(cls.nightly_features_all_file, 'r') as file:
            nightly_features_matlab = list(file['featfeat'])
        cls.nightly_features_matlab = np.array(nightly_features_matlab)
        with h5py.File(cls.channel_groups_file, 'r') as file:
            channel_groups_matlab = list(file['ch_gr'])
        cls.channel_groups_matlab = np.array(channel_groups_matlab)
        with h5py.File(cls.features_processed_all_file, 'r') as file:
            features_processed_matlab = list(file['feature'])
        cls.features_processed_matlab = np.transpose(np.array(features_processed_matlab), (0, 2, 1))
        with h5py.File(cls.sa_all_file, 'r') as file:
            sa_matlab = list(file['sa'])
        cls.sa_matlab = np.array(sa_matlab)
        with h5py.File(cls.mm_all_file, 'r') as file:
            mm_matlab = list(file['mm'])
        cls.mm_matlab = np.array(mm_matlab)
        with h5py.File(cls.postprob_all_file, 'r') as file:
            cls.postprob_matlab = np.array(list(file['postprob']))
        cls.sleepseeg_3min = SleepSEEG(filepath=cls.data_file_3min)
        cls.sleepseeg = SleepSEEG(filepath=cls.full_data_file)

        cls.matlab_model_import = MatlabModelImport(model_filepath=cls.model_filepath, gc_filepath=cls.gc_filepath)

    def test_get_epoch(self):
        epoch = self.sleepseeg.get_epoch_by_index(epoch_index=0)
        error = rmse(self.epoch_0_matlab_data, epoch.data[0])
        relative_rmse = error / (np.max(self.epoch_0_matlab_data) - np.min(self.epoch_0_matlab_data))
        self.assertTrue(relative_rmse < 0.01)

        diff = np.subtract(self.epoch_0_matlab_data, epoch.data[0])

        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].set_title("Epoch 1 LTP1 MATLAB")
        ax[0].plot(epoch._timestamps, self.epoch_0_matlab_data, linewidth=1)
        ax[0].set_ylabel('Signal amplitude [uV]')
        ax[1].set_title("Epoch 1 LTP1 Python")
        ax[1].plot(epoch._timestamps, epoch.data[0], linewidth=1)
        ax[1].set_xlabel('Timestamps [s]')
        ax[1].set_ylabel('Signal amplitude [uV]')
        plt.show()

        self.assertTrue(np.allclose(self.epoch_0_matlab_data, epoch.data[0], atol=0.5))

    def test_compute_features(self):
        # TODO: issue with compute features
        features = self.sleepseeg.extract_epochs_and_compute_features()
        for feat_idx in range(features.shape[0]):
            if not np.allclose(features[feat_idx], self.features_unprocessed_matlab[feat_idx], rtol=0.02):
                for ch_idx in range(features.shape[1]):
                    if not np.allclose(features[feat_idx, ch_idx], self.features_unprocessed_matlab[feat_idx, ch_idx], rtol=0.02):
                        print(feat_idx, ch_idx)

        self.assertTrue(np.allclose(features, self.features_unprocessed_matlab))

    def test_preprocess_features(self):
        self.sleepseeg_3min.features = self.features_unprocessed_matlab
        _, nightly_features_python = self.sleepseeg_3min.preprocess_features()
        self.assertTrue(np.allclose(nightly_features_python, self.nightly_features_processed_matlab, rtol=0.01))

    def test_cluster_channels(self):
        channel_groups_python = self.sleepseeg_3min.cluster_channels(
            nightly_features=self.nightly_features_processed_matlab,
            gc=self.matlab_model_import.GC
        )

        self.assertTrue(np.all(np.equal(channel_groups_python, self.channel_groups_matlab - 1)))

    def test_compute_probabilities(self):
        postprob_python = self.sleepseeg_3min._score_channels(models=self.matlab_model_import.models,
                                            features=self.features_processed_matlab,
                                                              channel_groups=self.channel_groups_matlab[0] - 1)
        postprob_matlab = np.transpose(self.postprob_matlab, (2, 1, 0))

        # Find the absolute difference between the arrays
        diff = np.abs(postprob_python - postprob_matlab)

        # Find the indices where the difference exceeds the tolerance
        mismatch_indices = np.where(diff > 0.01 * np.abs(postprob_matlab) + 1e-8)
        mismatch_indices_reshaped = list(zip(*mismatch_indices))

        self.assertTrue(np.allclose(postprob_python, postprob_matlab, rtol=0.5, atol=1e-8))

    def test_score_channels(self):
        # features shape (features x channels x epochs)
        sa, mm = self.sleepseeg_3min.score_epochs(features=self.features_processed_matlab,
                                                             models=self.matlab_model_import.models,
                                                             channel_groups=self.channel_groups_matlab[0] - 1)
        self.assertTrue(np.all(np.equal(sa, self.sa_matlab[0])))
        self.assertTrue(np.allclose(mm, self.mm_matlab[0], rtol=0.03))


class TestFeatures(unittest.TestCase):
    data_file_3min = r'eeg_data/auditory_stimulation_P18_002_3min.edf'
    features_3min_file = r'matlab_files/features_3min_bipolar.mat'
    features_3min_without_outliers_file = r'matlab_files/features_without_outliers_bipolar_3min.mat'
    features_smoothed_normalized_file = r'matlab_files/features_smoothed_normalized_v2_time.mat'
    features_unprocessed_all_file = r'matlab_files/features_unprocessed_v2_time.mat'
    features_processed_all_file = r'matlab_files/features_processed_v2_time.mat'
    nightly_features_all_file = r'matlab_files/nightly_features_v2_time.mat'

    @classmethod
    def setUpClass(cls) -> None:
        with h5py.File(cls.features_3min_file, 'r') as file:
            features_matlab = list(file['feature'])
        features_matlab = np.array(features_matlab)
        cls.features_matlab = np.transpose(features_matlab, (2, 1, 0))
        with h5py.File(cls.features_3min_without_outliers_file, 'r') as file:
            features_without_outliers_matlab = list(file['feature'])
        features_without_outliers_matlab = np.array(features_without_outliers_matlab)
        cls.features_without_outliers_matlab = np.transpose(features_without_outliers_matlab, (0, 2, 1))
        with h5py.File(cls.features_smoothed_normalized_file, 'r') as file:
            features_smoothed_matlab = list(file['feature'])
        features_smoothed_matlab = np.array(features_smoothed_matlab)
        cls.features_smoothed_matlab = np.transpose(features_smoothed_matlab, (0, 2, 1))
        with h5py.File(cls.features_unprocessed_all_file, 'r') as file:
            features_unprocessed_matlab = list(file['feature'])
        features_unprocessed_matlab = np.array(features_unprocessed_matlab)
        cls.features_unprocessed_matlab = np.transpose(features_unprocessed_matlab, (0, 2, 1))
        with h5py.File(cls.nightly_features_all_file, 'r') as file:
            nightly_features_matlab = list(file['featfeat'])
        cls.nightly_features_matlab = np.array(nightly_features_matlab)

        cls.sleepseeg_3min = SleepSEEG(filepath=cls.data_file_3min)
        cls.sleepseeg_3min.features = cls.features_unprocessed_matlab

    def test_remove_outliers(self):
        # TODO : this is wrong: I should use data that have outliers to make sure I remove them just like in matlab
        # I think I have fixed the issue though. Just make a better test this one is useless
        self.features = Features(input_array=self.features_matlab)
        features_without_outliers_python = self.features.remove_outliers()
        self.assertEqual(np.sum(np.isnan(features_without_outliers_python)),
                         np.sum(np.isnan(self.features_without_outliers_matlab)))

    def test_smooth_features(self):
        self.features = Features(input_array=self.features_unprocessed_matlab)
        features_smoothed_python = self.features.smooth_features()
        self.assertTrue(np.allclose(features_smoothed_python, self.features_smoothed_matlab, rtol=0.01))

    def test_get_nightly_features(self):
        self.features = Features(input_array=self.features_unprocessed_matlab)
        nightly_features_python = self.features.get_nightly_features()
        self.assertTrue(np.allclose(nightly_features_python, self.nightly_features_matlab, rtol=0.01))


class TestEpoch(unittest.TestCase):
    epoch_0_file = r'matlab_files/epoch_0_LTP1.mat'
    epoch_0_resampled = r'matlab_files/epoch_0_LTP1_resampled.mat'
    epoch_0_resampled_trimmed = r'matlab_files/epoch_0_LTP1_resampled_trimmed.mat'
    epoch_0_resampled_trimmed_baselined = r'matlab_files/epoch_0_LTP1_resampled_trimmed_baselined.mat'
    full_data_file = r'eeg_data/auditory_stimulation_P18_002.edf'
    features_file = r'matlab_files/features_epoch_0_chan_0_LTP1.mat'
    epoch_0_bipolar_chan1_file = r'matlab_files/epoch_0_bipolar_chan1.mat'
    epoch_0_ref_file = r'matlab_files/epoch_0_ref.mat'
    data_file_3min = r'eeg_data/auditory_stimulation_P18_002_3min.edf'
    epoch_0_bipolar_chan1_processed_file = r'matlab_files/epoch_0_bipolar_chan1_processed.mat'

    @classmethod
    def setUpClass(cls) -> None:
        with h5py.File(cls.epoch_0_file, 'r') as file:
            cls.epoch_0_matlab_data = list(file['x_save'])[0]
        with h5py.File(cls.epoch_0_resampled, 'r') as file:
            cls.epoch_0_matlab_data_resampled = list(file['x_save'])[0]
        with h5py.File(cls.epoch_0_resampled_trimmed, 'r') as file:
            cls.epoch_0_matlab_data_resampled_trimmed = list(file['x_save'])[0]
        with h5py.File(cls.epoch_0_resampled_trimmed_baselined, 'r') as file:
            cls.epoch_0_matlab_data_resampled_trimmed_baselined = list(file['x_save'])[0]
        with h5py.File(cls.features_file, 'r') as file:
            cls.features_epoch_0_LTP1_matlab = list(file['feature_0_1'])[0]
        with h5py.File(cls.epoch_0_bipolar_chan1_file, 'r') as file:
            cls.epoch_0_bipolar_chan1_matlab = list(file['x_save'])[0]
        # with h5py.File(cls.epoch_0_ref_file, 'r') as file:
        #     cls.epoch_0_ref_matlab_data = list(file['x_save'])
        with h5py.File(cls.epoch_0_bipolar_chan1_processed_file, 'r') as file:
            cls.epoch_0_bipolar_chan1_processed = list(file['x_to_save'])


    def test_change_sampling_rate(self):
        epoch_data = np.vstack([self.epoch_0_matlab_data, self.epoch_0_matlab_data])
        epoch = Epoch(data=epoch_data, timestamps=None, fs=2048, montage=None)
        epoch.change_sampling_rate()

        self.assertTrue(np.allclose(epoch.data[0, :], self.epoch_0_matlab_data_resampled, rtol=0.01))


    def test_trim(self):
        epoch = Epoch(data=self.epoch_0_matlab_data_resampled, timestamps=None, fs=256, montage=None)
        epoch.trim(n_samples_from_start=int(0.25*epoch.fs), n_samples_to_end=int(0.25*epoch.fs))

        self.assertTrue(np.allclose(epoch.data, self.epoch_0_matlab_data_resampled_trimmed, rtol=0.01))

    def test_mean_baseline(self):
        epoch_resampled_trimmed_baselined_python = Epoch(data=self.epoch_0_matlab_data_resampled_trimmed,
                                                         timestamps=None,
                                                         fs=256,
                                                         montage=None)
        epoch_resampled_trimmed_baselined_python.mean_normalize()
        self.assertTrue(np.allclose(epoch_resampled_trimmed_baselined_python.data, self.epoch_0_matlab_data_resampled_trimmed_baselined, rtol=0.01))

    def test_compute_signal_features(self):
        epoch = Epoch(data=self.epoch_0_matlab_data_resampled_trimmed_baselined, timestamps=None, fs=256, montage=None)
        features_python = epoch._compute_signal_features(data=epoch.data)

        self.assertTrue(np.allclose(features_python, self.features_epoch_0_LTP1_matlab, rtol=0.01))

    # def test_make_bipolar_montage(self):
    #
    #     edf = EdfReader(filepath=self.data_file_3min)
    #     edf.clean_channel_names()
    #     montage = edf.get_montage()
    #     epoch_python = Epoch(data=np.vstack(self.epoch_0_ref_matlab_data), timestamps=None, fs=2048, montage=montage)
    #     epoch_python.make_bipolar_montage()
    #
    #     self.assertTrue(np.allclose(self.epoch_0_bipolar_chan1_matlab, epoch_python.data[0], rtol=0.01))

    # def test_preprocess_epoch(self):
    #     epoch_python = Epoch(data=self.epoch_0_bipolar_chan1_matlab, fs=2048, timestamps=None)
    #     epoch_python.change_sampling_rate()
    #     epoch_python.trim(n_samples_from_start=int(0.25 * epoch_python.fs), n_samples_to_end=int(0.25 * epoch_python.fs))
    #     epoch_python.mean_normalize()
    #
    #     difference = np.subtract(self.epoch_0_bipolar_chan1_processed[0], epoch_python.data)
    #     rel_dif = np.divide(difference, self.epoch_0_bipolar_chan1_processed[0])
    #
    #     fig, ax1 = plt.subplots()
    #
    #     color = 'tab:red'
    #     ax1.set_xlabel('samples')
    #     ax1.set_ylabel('absolute difference with python', color=color)
    #     ax1.plot(difference, color=color, linestyle=':')
    #     ax1.tick_params(axis='y', labelcolor=color)
    #
    #     ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    #
    #     color = 'tab:blue'
    #     ax2.set_ylabel('matlab processed channel 0', color=color)  # we already handled the x-label with ax1
    #     ax2.plot(self.epoch_0_bipolar_chan1_processed[0], color=color, alpha = 0.3)
    #     ax2.plot(epoch_python.data, color=color, alpha = 0.3)
    #     ax2.tick_params(axis='y', labelcolor=color)
    #
    #     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #     plt.show()
    #
    #     error = rmse(epoch_python.data, self.epoch_0_bipolar_chan1_processed[0])
    #     relative_rmse = error / (np.max(self.epoch_0_bipolar_chan1_processed[0]) - np.min(
    #         self.epoch_0_bipolar_chan1_processed[0]))
    #
    #     self.assertTrue(np.allclose(self.epoch_0_bipolar_chan1_processed[0], epoch_python.data, rtol=0.1))

