import logging
import typing as t
import os
from datetime import datetime, timedelta

from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.features import Features
from models.epoch import Epoch
from models.layout import Montage
from models.readers.reader_factory import EEGReaderFactory
from models.MatlabModelImport import MatlabModelImport, ClassificationTree


STAGENAMES = {1: 'R', 2: 'W', 3: 'N1', 4: 'N2', 5: 'N3'}


class SleepSEEG:
    def __init__(self, filepath: str):
        self.logger = self._init_logger()
        self._filepath = filepath
        self._edf = EEGReaderFactory.get_reader(self._filepath)
        self.montage_type = 'referential'
        self.epochs = []
        self.sleep_stages = None
        self.summary = None
        self.features = None
        self._initial_buffer = None
        self._time_window = None
        self.sleep_stage = None

    def _init_logger(self):
        logger = logging.getLogger('SleepSEEG_logger')
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def get_epoch_by_index(self, epoch_index: int):
        rolling_window_seconds = 2.5
        epoch_zero_start = int(self._edf.start_time_sample - rolling_window_seconds / 2 * self._edf.sampling_frequency)
        # epoch_zero_start = int(self._edf.start_time_sample)

        epoch_start = int(epoch_zero_start + epoch_index * 30 * self._edf.sampling_frequency)
        epoch_end = (epoch_zero_start + (epoch_index + 1) * 30 * self._edf.sampling_frequency
                     + rolling_window_seconds * self._edf.sampling_frequency)

        epoch = self.read_epoch(start_sample=epoch_start, end_sample=epoch_end)
        epoch.start_time = self._edf.start_timenum + epoch_index / 28800
        epoch.start_sample = epoch_start
        return epoch

    def compute_epoch_features(self) -> Features:
        self.logger.info("Started epoch extraction and feature computation.")
        features_list = [self._process_epoch(idx) for idx in tqdm(range(self._edf.n_epochs), "Computing features")]
        self.features = Features(input_array=np.array(features_list).transpose(2, 1, 0))
        return self.features

    def _process_epoch(self, epoch_index: int):
        epoch = self.get_epoch_by_index(epoch_index)
        epoch.make_bipolar_montage()
        epoch.change_sampling_rate_deepseek()
        epoch.trim(n_samples_from_start=int(0.25 * epoch.fs), n_samples_to_end=int(0.25 * epoch.fs))
        epoch.mean_normalize()
        self.epochs.append(epoch)
        return epoch.compute_features()

    def preprocess_features(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features. Features should be an array of shape (Nfeat, Nchans, Nepochs)
        :param features:
        :return:
        """

        # Outlier detection: remove outliers comparing the same feature on same channel comparing across epoch axis
        self.features.remove_outliers()

        # Smoothing & Normalizing
        self.features.smooth_features()

        # Extract nightly features
        nightly_features = self.features.get_nightly_features()

        return self.features, nightly_features

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

    def cluster_channels(self, nightly_features: np.ndarray, gc):
        """
        This is a function that does this and that. BLoup.

        :param nightly_features:
        :param gc:
        :return:
        """
        Nch = nightly_features.shape[1]
        channel_groups = np.zeros(Nch, dtype=int)
        Ng = gc.shape[0]

        for nc in range(Nch):
            differences = np.tile(nightly_features[:, nc], (Ng, 1)) - gc
            distances = np.sum(differences ** 2, axis=1)  # Compute squared Euclidean distances
            channel_groups[nc] = np.argmin(distances)  # Find index of minimum distance

        return channel_groups

    def _score_channels(self, models, features, channel_groups):
        n_features, n_channels, n_epochs = features.shape
        n_stages = 7
        feat = np.transpose(features, (1, 2, 0))
        postprob = np.zeros(shape=(n_channels, n_epochs, n_stages))
        clusters = np.unique(channel_groups).astype(int)
        self.logger.info(msg=f"{len(clusters)} clusters found.")
        for cluster_idx in tqdm(clusters, "Computing probabilities for cluster"):
            ik = channel_groups == cluster_idx
            if np.any(ik):
                fe = feat[ik, :, :].reshape(np.count_nonzero(ik) * n_epochs, n_features, order='F')

                del_rows = np.all(np.isnan(fe), axis=1)
                prob = np.full((fe.shape[0], 7), np.nan)

                # Predict using each model
                prob[~del_rows, :4] = models[cluster_idx][0].predict_proba_deepseek(fe[~del_rows])
                prob[~del_rows, 4] = models[cluster_idx][1].predict_proba_deepseek(fe[~del_rows])[:, 0]
                prob[~del_rows, 5] = models[cluster_idx][2].predict_proba_deepseek(fe[~del_rows])[:, 0]
                prob[~del_rows, 6] = models[cluster_idx][3].predict_proba_deepseek(fe[~del_rows])[:, 0]

                prob_reshaped = prob.reshape(np.count_nonzero(ik), n_epochs, 7, order='F')
                postprob[np.where(ik)[0], :, :] = prob_reshaped
        return postprob

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
            confidence = np.nanmean(prob, axis=0)  # Average over channels
            confidence[:, :4] /= np.sum(confidence[:, :4], axis=1, keepdims=True)  # Normalize first 4 columns
            return confidence

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

        postprob = self._score_channels(models=models, features=features, channel_groups=channel_groups)
        confidence = _compute_confidence(prob=postprob)
        sa, mm = _define_stage(confidence=confidence)

        for i, epoch in enumerate(self.epochs):
            epoch.stage = sa[i]
            epoch.max_confidence = mm[i]

        self.logger.info(msg="Finished scoring epochs.")
        return sa, mm

    def export_sleep_stage_output(self, output_folder: str, filename: str = 'SleepStage.csv'):
        # Sort epochs per start time
        self.epochs.sort(key=lambda x:x.start_time)

        file_indentifiers = np.ones(shape=len(self.epochs))
        epoch_start_times = [epoch.start_time for epoch in self.epochs]
        stages = [epoch.stage for epoch in self.epochs]
        max_confidences = [epoch.max_confidence for epoch in self.epochs]
        epoch_start_samples = [epoch.start_sample for epoch in self.epochs]

        self.sleep_stages = pd.DataFrame({
            'FileIdentifier': file_indentifiers,
            'EpochStartTime': epoch_start_times,
            'SleepStage': stages,
            'MaximumConfidence': max_confidences,
            'EpochStartSample': epoch_start_samples
        })

        self.sleep_stages.to_csv(os.path.join(output_folder, filename), index=False)

        return self.sleep_stages

    def export_summary_output(self, output_folder: str, filename: str = 'Summary.csv'):

        def _matlab_datenum_to_datetime(datenum):
            return datetime(1, 1, 1) + timedelta(days=datenum)

        # Identify file transitions or stage transitions between successive epochs
        ic = (
                (self.sleep_stages['SleepStage'] != self.sleep_stages['SleepStage'].shift()) |  # Sleep stage changes
                (self.sleep_stages['FileIdentifier'] != self.sleep_stages['FileIdentifier'].shift())  # File identifier changes
        )
        # Add a True at the beginning to mark the first epoch as a transition
        ic = pd.concat([pd.Series([True]), ic[1:]]).reset_index(drop=True)

        # Extract rows where transitions occur
        Sl = self.sleep_stages[ic].reset_index(drop=True)

        # Get epoch count between each transition
        icc = np.where(ic)[0]
        icc = np.append(icc, len(ic))  # Add dummy index
        icc = icc[1:] - icc[:-1]  # Calculate epoch counts

        n_groups = Sl.shape[0]
        file_list = [self._filepath.split('/')[-1]] * n_groups

        sleep_stage_list = []
        n_epochs_list = []
        date_list = []
        time_list = []
        for i in range(n_groups):
            sleep_stage_list.append(STAGENAMES[Sl.loc[i, 'SleepStage']])
            n_epochs_list.append(icc[i])

            dt = _matlab_datenum_to_datetime(Sl.loc[i, 'EpochStartTime'])
            date_list.append(dt.strftime('%Y-%m-%d'))
            time_list.append(dt.strftime('%H:%M:%S'))

        self.summary = pd.DataFrame({
            'File': file_list,
            'Date': date_list,
            'Time': time_list,
            'Sleep stage': sleep_stage_list,
            '# of epochs': n_epochs_list
        })

        self.summary.to_csv(os.path.join(output_folder, filename), index=False)

        return self.summary



if __name__ == "__main__":
    filepath = r'../eeg_data/auditory_stimulation_P18_002_3min.edf'
    sleep_eeg_instance = SleepSEEG(filepath=filepath)

    #cProfile.run('sleep_eeg_instance.compute_epoch_features()')
    features = sleep_eeg_instance.compute_epoch_features()
    features_preprocessed, nightly_features = sleep_eeg_instance.preprocess_features(features=features)

    parameters_directory = r'../model_parameters'
    model_filename = r'Model_BEA_full.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)

    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)
    matlab_model_import = MatlabModelImport(model_filepath=model_filepath, gc_filepath=gc_filepath)

    channel_groups = sleep_eeg_instance.cluster_channels(nightly_features=nightly_features,
                                                         gc=matlab_model_import.GC)
    sa, mm = sleep_eeg_instance.score_epochs(features=features_preprocessed,
                                              models=matlab_model_import.models,
                                              channel_groups=channel_groups)

    sleep_stage = sleep_eeg_instance.export_sleep_stage_output(output_folder=r'../results', filename='SleepStage_3min.csv')
    summary = sleep_eeg_instance.export_summary_output(output_folder=r'../results', filename='Summary_3min.csv')