import os
import unittest
import numpy as np
import tempfile
import shutil

import pyedflib

from models.readers.edf_reader import EdfReader

if os.getcwd().split('\\')[-1] == 'test':
    os.chdir("..")  # Go one directory up


def create_sine_wave_edf(filepath, freq=10, duration=1, sampling_rate=512, amplitude=100):
    """
    Creates an EDF file with a single-channel sinusoidal wave.

    - freq: Frequency of the sine wave in Hz
    - duration: Length of the signal in seconds
    - sampling_rate: Number of samples per second
    - amplitude: Amplitude of the sine wave
    """
    num_samples = duration * sampling_rate
    time = np.linspace(0, duration, num_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * time)

    digital_min = -32768
    digital_max = 32767
    signal_headers = [{
        'label': 'EEG Fz-Cz',
        'dimension': 'uV',
        'sample_frequency': sampling_rate,
        'physical_min': digital_min,
        'physical_max': digital_max,
        'digital_min': digital_min,
        'digital_max': digital_max,
        'transducer': '',
        'prefilter': ''
    }]

    signal_scaled = np.round((signal / amplitude) * digital_max).astype(np.int32)

    with pyedflib.EdfWriter(filepath, 1, file_type=pyedflib.FILETYPE_EDF) as f:
        f.setSignalHeaders(signal_headers)
        # f.writePhysicalSamples(signal)
        f.writeDigitalSamples(signal_scaled)

    return filepath, signal_scaled, sampling_rate

class TestEdfReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.temp_folder = tempfile.TemporaryDirectory()
        edf_file = "test.edf"
        edf_path = os.path.join(cls.temp_folder.name, edf_file)
        filepath, expected_signal, sampling_rate = create_sine_wave_edf(str(edf_path))
        cls.edf = EdfReader(filepath=filepath)
        cls.expected_signal = expected_signal

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

        shutil.rmtree(cls.temp_folder.name)

    def test_extract_data_pyedflib(self):
        data, _ = self.edf.extract_data_pyedflib(start_sample=0, end_sample=512, digital=True)
        self.assertTrue(np.array_equal(self.expected_signal, data))

class TestEdfMNEReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.temp_folder = tempfile.TemporaryDirectory()
        edf_file = "test.edf"
        edf_path = os.path.join(cls.temp_folder.name, edf_file)
        filepath, expected_signal, sampling_rate = create_sine_wave_edf(str(edf_path))
        cls.edf = EdfReader(filepath=filepath)
        cls.expected_signal = expected_signal

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

        shutil.rmtree(cls.temp_folder.name)

    def test_extract_data_mne(self):
        data, _, _ = self.edf.load_data(start_sample=0, end_sample=512)
        self.assertTrue(np.allclose(self.expected_signal, data, rtol=0.01))