import os
import unittest
import numpy as np
import h5py

from eeg_reader import EdfReader

if os.getcwd().split('\\')[-1] == 'test':
    os.chdir("..")  # Go one directory up

class TestEdfReader(unittest.TestCase):
    epoch_0_file = r'matlab_files/epoch_0_LTP1.mat'
    full_data_file = r'eeg_data/auditory_stimulation_P18_002.edf'

    @classmethod
    def setUpClass(cls) -> None:
        with h5py.File(cls.epoch_0_file, 'r') as file:
            cls.epoch_0_matlab_data = list(file['x_save'])[0]
        cls.edf = EdfReader(filepath=cls.full_data_file)

    def test_extract_data_pyedflib(self):
        window_center = self.edf.start_time_sample
        window_length_seconds = 2.5
        window_length_samples = window_length_seconds * self.edf.sampling_frequency
        start_sample = window_center - round(window_length_samples / 2)
        stop_sample = window_center + round(window_length_samples / 2 + 30 * self.edf.sampling_frequency)
        data, _ = self.edf.extract_data_pyedflib(start_sample=start_sample, stop_sample=stop_sample)

        def rmse(signal1, signal2):
            return np.sqrt(np.mean((signal1 - signal2) ** 2))

        error = rmse(self.epoch_0_matlab_data, data[0])
        relative_rmse = error / (np.max(self.epoch_0_matlab_data) - np.min(self.epoch_0_matlab_data))

        self.assertTrue(relative_rmse < 0.01)