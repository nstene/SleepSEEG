import os
import unittest
import numpy as np
import h5py

from main import run_analysis
from models.sleep_seeg import SleepSEEG


if os.getcwd().split('\\')[-1] == 'test':
    os.chdir("..")  # Go one directory up

class TestEdfReader(unittest.TestCase):
    sleepstage_matlab_file = r'results/SleepStage_full.mat'
    full_data_file = 'auditory_stimulation_P18_002.edf'

    @classmethod
    def setUpClass(cls) -> None:
        with h5py.File(cls.sleepstage_matlab_file, 'r') as file:
            cls.sleepstage_matalb = list(file['SleepStage'])

    def test_run_analysis(self):
        sleepstage, summary = run_analysis(filename=self.full_data_file, automatic=True)

        stages_python = np.array(sleepstage.iloc[:, 2])
        stages_matlab = self.sleepstage_matalb[2].astype(int)

        n_diff = len(np.where(stages_python != stages_matlab))
        scoring_accuracy = n_diff/len(stages_matlab)

        confidence_python = np.array(sleepstage.iloc[:, 3])
        confidence_matlab = np.round(self.sleepstage_matalb[3], 4)

        diff = np.subtract(confidence_python, confidence_matlab)
        rel_diff = np.divide(diff, confidence_matlab)

        n_epochs = len(stages_python)

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].set_title("Comparison of sleep stages per epoch")
        ax[0].scatter(x=np.linspace(1, n_epochs, n_epochs).astype(int), y=stages_python, label="Python", color="blue", alpha=0.7)
        ax[0].scatter(x=np.linspace(1, n_epochs, n_epochs).astype(int), y=stages_matlab, label="Matlab", color="red", alpha=0.7)
        ax[0].set_ylabel('Sleep stage')
        ax[0].set_yticks([1, 2, 3, 4, 5])
        ax[0].grid(True, linestyle="--", alpha=0.5)
        ax[0].legend()

        ax[1].set_title("Difference in sleep score maximum confidence")
        ax[1].plot(diff, marker='o', markersize=1)
        ax[1].set_ylabel('Delta')
        ax[1].set_xlabel('Epochs')
        ax[1].set_xticks(np.arange(1, n_epochs + 1, 10))
        ax[1].grid(True, linestyle="--", alpha=0.5)
        plt.show()

        print('hi')


