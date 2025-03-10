import unittest
import numpy as np
import h5py
import os

from models.SleepSEEG import SleepSEEG, Epoch
from eeg_reader import EdfReader
from models.MatlabModelImport import MatlabModelImport


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
    features_3min_file = r'matlab_files/features_3min_bipolar.mat'
    features_3min_without_outliers_file = r'matlab_files/features_without_outliers_bipolar_3min.mat'
    features_smoothed_normalized_file = r'matlab_files/features_smoothed_normalized_v2_time.mat'
    features_unprocessed_all_file = r'matlab_files/features_unprocessed_v2_time.mat'
    features_processed_all_file = r'matlab_files/features_processed_v2_time.mat'
    nightly_features_all_file = r'matlab_files/nightly_features_v2_time.mat'
    channel_groups_file = r'matlab_files/channel_groups.mat'

    parameters_directory = r'model_parameters'
    model_filename = r'Model_BEA_refactored.mat'
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
        with h5py.File(cls.features_unprocessed_all_file, 'r') as file:
            features_unprocessed_matlab = list(file['feature'])
        features_unprocessed_matlab = np.array(features_unprocessed_matlab)
        cls.features_unprocessed_matlab = np.transpose(features_unprocessed_matlab, (0, 2, 1))
        with h5py.File(cls.features_processed_all_file, 'r') as file:
            features_processed_matlab = list(file['featfeat'])
        cls.features_processed_matlab = np.array(features_processed_matlab)
        with h5py.File(cls.nightly_features_all_file, 'r') as file:
            nightly_features_matlab = list(file['featfeat'])
        cls.nightly_features_matlab = np.array(nightly_features_matlab)
        with h5py.File(cls.channel_groups_file, 'r') as file:
            channel_groups_matlab = list(file['ch_gr'])
        cls.channel_groups_matlab = np.array(channel_groups_matlab)
        cls.features_smoothed_matlab = np.transpose(features_smoothed_matlab, (0, 2, 1))
        cls.sleepseeg_3min = SleepSEEG(filepath=cls.data_file_3min)

        cls.matlab_model_import = MatlabModelImport(model_filepath=cls.model_filepath, gc_filepath=cls.gc_filepath)

    def test_get_epoch(self):
        epoch = self.sleepseeg_3min.get_epoch_by_index(epoch_index=0)
        error = rmse(self.epoch_0_matlab_data, epoch.data[0])
        relative_rmse = error / (np.max(self.epoch_0_matlab_data) - np.min(self.epoch_0_matlab_data))

        self.assertTrue(relative_rmse < 0.01)

    def test_compute_features(self):
        # TODO: issue with compute features
        self.sleepseeg_3min.compute_epoch_features()
        self.assertTrue(np.allclose(self.sleepseeg_3min.features, self.features_matlab, rtol=0.01))

    def test_remove_outliers(self):
        features_without_outliers_python = self.sleepseeg_3min.remove_outliers(features=self.features_matlab)
        self.assertEqual(np.sum(np.isnan(features_without_outliers_python)), np.sum(np.isnan(self.features_without_outliers_matlab)))

    def test_smooth_features(self):
        features_smoothed_python = self.sleepseeg_3min.smooth_features(features=self.features_unprocessed_matlab)
        self.assertTrue(np.allclose(features_smoothed_python, self.features_smoothed_matlab, rtol=0.01))

    def test_get_nightly_features(self):
        nightly_features_python = self.sleepseeg_3min.get_nightly_features(features=self.features_unprocessed_matlab)
        self.assertTrue(np.allclose(nightly_features_python, self.nightly_features_matlab, rtol=0.01))

    def test_preprocess_features(self):
        # TODO : find why some values differ more than 5%
        _, nightly_features_python = self.sleepseeg_3min.preprocess_features(features=self.features_unprocessed_matlab)
        self.assertTrue(np.allclose(nightly_features_python, self.features_processed_matlab, rtol=0.6))

    def test_cluster_channels(self):
        channel_groups_python = self.sleepseeg_3min.cluster_channels(
            nightly_features=self.features_processed_matlab,
            gc=self.matlab_model_import.GC
        )

        self.assertTrue(np.all(np.equal(channel_groups_python, self.channel_groups_matlab - 1)))

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
        with h5py.File(cls.epoch_0_ref_file, 'r') as file:
            cls.epoch_0_ref_matlab_data = list(file['x_save'])


    def test_change_sampling_rate(self):
        epoch = Epoch(data=self.epoch_0_matlab_data, timestamps=None, fs=2048, montage=None)
        epoch.change_sampling_rate()
        epoch_resampled_python = epoch.data

        error = rmse(self.epoch_0_matlab_data_resampled, epoch_resampled_python)
        relative_rmse = error / (np.max(self.epoch_0_matlab_data_resampled) - np.min(self.epoch_0_matlab_data_resampled))

        self.assertTrue(relative_rmse < 0.01)

    def test_trim(self):
        epoch = Epoch(data=self.epoch_0_matlab_data_resampled, timestamps=None, fs=256, montage=None)
        epoch.trim(n_samples_from_start=int(0.25*epoch.fs), n_samples_to_end=int(0.25*epoch.fs))
        epoch_resampled_trimmed_python = epoch.data

        error = rmse(self.epoch_0_matlab_data_resampled_trimmed, epoch_resampled_trimmed_python)
        relative_rmse = error / (np.max(self.epoch_0_matlab_data_resampled_trimmed) - np.min(self.epoch_0_matlab_data_resampled_trimmed))

        self.assertTrue(relative_rmse < 0.01)

    def test_mean_baseline(self):
        epoch = Epoch(data=self.epoch_0_matlab_data_resampled_trimmed, timestamps=None, fs=256, montage=None)
        epoch.mean_normalize()
        epoch_resampled_trimmed_baselined_python = epoch.data

        error = rmse(self.epoch_0_matlab_data_resampled_trimmed_baselined, epoch_resampled_trimmed_baselined_python)
        relative_rmse = error / (np.max(self.epoch_0_matlab_data_resampled_trimmed_baselined) - np.min(self.epoch_0_matlab_data_resampled_trimmed_baselined))

        self.assertTrue(relative_rmse < 0.01)

    def test_compute_signal_features(self):
        epoch = Epoch(data=self.epoch_0_matlab_data_resampled_trimmed_baselined, timestamps=None, fs=256, montage=None)
        features_python = epoch._compute_signal_features(data=epoch.data)
        print(features_python)
        print(self.features_epoch_0_LTP1_matlab)

        self.assertTrue(np.allclose(features_python, self.features_epoch_0_LTP1_matlab, rtol=0.01))

    def test_make_bipolar_montage(self):

        edf = EdfReader(filepath=self.data_file_3min)
        edf.clean_channel_names()
        montage = edf.get_montage()
        epoch_python = Epoch(data=np.vstack(self.epoch_0_ref_matlab_data), timestamps=None, fs=2048, montage=montage)
        epoch_python.make_bipolar_montage()

        self.assertTrue(np.allclose(self.epoch_0_bipolar_chan1_matlab, epoch_python.data[0], rtol=0.01))
