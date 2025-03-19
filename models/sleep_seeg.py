import logging
import typing as t
import os
from datetime import datetime, timedelta

from tqdm import tqdm
import numpy as np
import pandas as pd
import tkinter as tk

from models.features import Features
from models.epoch import Epoch
from models.layout import Montage, Channel
from models.readers.reader_factory import EEGReaderFactory
from models.matlab_adaptator import MatlabModelImport, ClassificationTree


STAGENAMES = {1: 'R', 2: 'W', 3: 'N1', 4: 'N2', 5: 'N3'}
SECONDS_IN_DAY = 60*60*24


class SleepSEEG:
    """A class for automatic scoring of Sleep iEEG recordings.

    This class provides functionality to read EEG data, extract epochs, compute features,
    and classify sleep stages using machine learning models. It also supports exporting
    results to CSV files.

    Attributes:
        logger (logging.Logger): Logger for the SleepSEEG module.
        _filepath (str): Path to the EDF file containing the EEG data.
        _edf (EEGReader): EEG reader object for reading and processing the EDF file.
        montage_type (str): Type of montage used for the EEG channels (default: 'referential').
        epochs (list[Epoch]): List of extracted epochs.
        sleep_stages (pd.DataFrame): DataFrame containing sleep stage classifications.
        summary (pd.DataFrame): Summary of sleep stages and transitions.
        features (Features): Computed features for each epoch.
        _initial_buffer (any): Placeholder for initial buffer data.
        _time_window (any): Placeholder for time window data.
        sleep_stage (any): Placeholder for sleep stage data.
        excluded_channels (list[Channel]): List of channels excluded from analysis.
    """

    _EPOCH_EXTRACTION_ROLLING_WINDOW = 30

    def __init__(self, filepath: str):
        """Initializes the SleepSEEG object.

        :param filepath: Path to the EDF file containing the EEG data.
        """
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
        self.excluded_channels = None

    @property
    def channel_names(self) -> t.List[str]:
        """Returns a list of the standardized channel names of the recording."""
        if not self._edf.channels:
            self._edf.clean_channel_names()

        channel_names = [chan.name for chan in self._edf.channels]
        return channel_names

    @staticmethod
    def _init_logger() -> logging.Logger:
        """Initialized the logger for the Sleep SEEG module.

        :return: logger
        """
        logger = logging.getLogger('SleepSEEG_logger')
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def get_epoch_by_index(self, epoch_index: int) -> Epoch:
        """Retrieves an epoch by its index.

        :param epoch_index: Index of the epoch to retrieve.
        :returns: The extracted epoch object.
        """
        rolling_window_seconds = 2.5
        epoch_zero_start = int(self._edf.start_time_sample - rolling_window_seconds / 2 * self._edf.sampling_frequency)

        epoch_start = int(epoch_zero_start + epoch_index * 30 * self._edf.sampling_frequency)
        epoch_end = (epoch_zero_start + (epoch_index + 1) * 30 * self._edf.sampling_frequency
                     + rolling_window_seconds * self._edf.sampling_frequency)

        epoch = self.read_epoch(start_sample=epoch_start, end_sample=epoch_end)
        epoch.start_time = self._edf.start_datetime + timedelta(0, epoch_index * 30)
        epoch.start_sample = epoch_start
        epoch.matlab_start_sample = int(self._edf.start_time_sample + epoch_index * 30 * self._edf.sampling_frequency)

        epoch.drop_channels(self.excluded_channels)

        return epoch

    def extract_epochs_and_compute_features(self, epoch_indices: t.List[int]=None, keep_epoch_data: bool = False) \
            -> Features:
        """Extracts epochs, preprocesses them and computes features for each of them.

        :param epoch_indices: List of epoch indices to extract. If None, all epochs are extracted.
        :param keep_epoch_data: Whether to retain the raw data in the epoch objects. Defaults to False.
        :return: Computed features for all epochs in the form (Nfeat, Nchans, Nepochs).
        """
        epochs_to_extract = epoch_indices if epoch_indices else range(self._edf.n_epochs)
        features_list = []
        epochs_list = []
        for idx in tqdm(epochs_to_extract, "Extracting epochs"):
            # Read epoch
            epoch = self.get_epoch_by_index(idx)

            # Preprocess epoch
            epoch.process_epoch()

            # Compute features
            features_list.append(epoch.compute_features())

            if not keep_epoch_data:
                epoch.set_data(data=None)
            epochs_list.append(epoch)

        self.features = Features(input_array=np.array(features_list).transpose(2, 1, 0))
        self.epochs = epochs_list
        return self.features

    def preprocess_features(self) -> t.Tuple[Features, np.ndarray]:
        """Preprocesses the computed features and computes the aggregated nightly features.
        Features are in the form (Nfeat, Nchans, Nepochs)

        :return: tuple(processed features (Nfeat, Nchans, Nepochs), aggregated nightly features (Nfeat, Nchans))
        """
        if self.features.shape[2] < 2:
            return self.features, np.squeeze(self.features)

        # Outlier detection: remove outliers comparing the same feature on same channel comparing across epoch axis
        self.features.remove_outliers()

        # Smoothing & Normalizing
        self.features.smooth_features()

        # Extract nightly features
        nightly_features = self.features.get_nightly_features()

        return self.features, nightly_features

    def read_epoch(self, start_sample: int, end_sample: int = None, epoch_duration: float = None) -> Epoch:
        """Reads an epoch from the EEG data using MNE.

        :param start_sample: Start sample index for the epoch.
        :param end_sample: End sample index for the epoch. If None, it is calculated using epoch_duration.
        :param epoch_duration: Duration of the epoch in seconds. Required if end_sample is None.

        :return: The extracted epoch object.
        """
        if not end_sample:
            end_sample = int(start_sample + epoch_duration * self._edf.sampling_frequency)

        if (end_sample % 1 != 0) or (start_sample % 1 != 0):
            raise ValueError(f"Start and end samples must be integers. "
                             f"start_sample:{start_sample}; end_sample: {end_sample}")

        start_sample = int(start_sample)
        end_sample = int(end_sample)

        data, timestamps, channels = self._edf.load_data(start_sample=start_sample, end_sample=end_sample)
        #data, channels = self._edf.extract_data_pyedflib(start_sample=start_sample, end_sample=end_sample, digital=True)
        montage = Montage(channels=channels, montage_type='referential')

        return Epoch(data=data, fs=self._edf.sampling_frequency, timestamps=timestamps, montage=montage)
        # return Epoch(data=data, fs=self._edf.sampling_frequency, timestamps=None, montage=montage)

    @staticmethod
    def cluster_channels(nightly_features: np.ndarray, gc: np.ndarray) -> np.ndarray:
        """Assigns each EEG channel to a cluster based on the nightly features.

        Clustering is done by comparing the nightly features of each channel to a set of pre-defined cluster centroids
        gc and assigning the channel to the closest cluster.

        :param nightly_features: Nightly features for clustering (Nfeat x Nchans).
        :param gc: Array (Nclusters x Nfeat).

        :return: Array of cluster assignments for each channel (Nchans).
        """
        Nch = nightly_features.shape[1]
        channel_groups = np.zeros(Nch, dtype=int)
        Ng = gc.shape[0]

        for nc in range(Nch):
            differences = np.tile(nightly_features[:, nc], (Ng, 1)) - gc
            distances = np.sum(differences ** 2, axis=1)
            channel_groups[nc] = np.argmin(distances)  # Find index of minimum distance

        return channel_groups

    def _score_channels(self, models: t.List[t.List[ClassificationTree]], features: Features,
                        channel_groups: np.ndarray) -> np.ndarray:
        """Scores channels using machine learning models.

        The method uses pre-trained classification trees to predict the probability that each epoch belongs to each
        sleep stage (we use 7 of them).

        :param models: List of machine learning models (ClassificationTree) for each cluster (Ncluster x 4).
        :param features: Features to score (Nfeat x Nchans x Nepochs).
        :param channel_groups: Cluster assignments for each channel (Nchans).

        :return: Posterior probabilities for each channel and epoch (Nchans x Nepochs x Nstages).
        """
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
                prob[~del_rows, :4] = models[cluster_idx][0].predict_proba(fe[~del_rows])
                prob[~del_rows, 4] = models[cluster_idx][1].predict_proba(fe[~del_rows])[:, 0]
                prob[~del_rows, 5] = models[cluster_idx][2].predict_proba(fe[~del_rows])[:, 0]
                prob[~del_rows, 6] = models[cluster_idx][3].predict_proba(fe[~del_rows])[:, 0]

                prob_reshaped = prob.reshape(np.count_nonzero(ik), n_epochs, 7, order='F')
                postprob[np.where(ik)[0], :, :] = prob_reshaped
        return postprob

    def score_epochs(self, features: Features, models: t.List[t.List[ClassificationTree]], channel_groups: np.ndarray) \
            -> t.Tuple[np.ndarray, np.ndarray]:
        """Scores epochs using classification trees.

        For each cluster group, find channels that belong to that group
        Extract their features, reshape them, remove rows that have only NaNs
        Apply the four machine learning models to predict probabilities prob
        Compute average posterior probabilities: output is confidence, the normalized probabilities

        :param models: List of machine learning models (ClassificationTree) for each cluster (Ncluster x 4).
        :param features: Features to score (Nfeat x Nchans x Nepochs).
        :param channel_groups: Cluster assignments for each channel (Nchans).

        :return: tuple(sleep stages (Nepochs), maximum confidence (Nepochs))
        """

        def _compute_confidence(postprob: np.ndarray):
            """Computes the final confidence matrix by averaging across channels and normalizing.

            :param postprob: Posterior probabilities for each channel and epoch (Nchans x Nepochs x Nstages).

            :return: normalized confidences (Nepochs x Nstages), the likelihood that each epoch belongs to each stage
            """
            confidence = np.nanmean(postprob, axis=0)  # Average over channels
            confidence[:, :4] /= np.sum(confidence[:, :4], axis=1, keepdims=True)  # Normalize first 4 columns
            return confidence

        def _define_stage(confidence):
            """Defines sleep stages based on confidence values.

            Confidence values from first to fourth columns give likelihoods that epoch is in stage R, W, N2 and N3
            respectively.

            Confidence value in fifth, sixth and seventh columns give likelihoods to discriminate stage N1 to R, W and
            N2 respectively. N1 is not compared to N3 as usually when an epoch has been scored N3 it is very unlikely
            that the ground truth is N1.

            To match the confidence indices with the stages dictionary defined in the code, we shift stages N2 and N3
            by a unit, then stage N1 can be assigned using the value 3.

            Nans are identified with stage 2.
            """
            # Define stage for each epoch
            mm = np.max(confidence[:, :4], axis=1)
            sa = np.argmax(confidence[:, :4], axis=1) + 1  # stage index

            # handle nans
            sa[np.isnan(mm)] = 2
            mm[np.isnan(mm)] = 0

            # Shift indices to match the dictionary's
            sa[sa > 2] += 1

            # Adjust stage to discriminate N1 and R, W and N2, respectively.
            sa[(sa == 1) & (confidence[:, 4] < 0.5)] = 3 # N1 and R
            sa[(sa == 2) & (confidence[:, 5] < 0.5)] = 3 # N1 and W
            sa[(sa == 4) & (confidence[:, 6] < 0.5)] = 3 # N1 and N2

            return sa, mm

        postprob = self._score_channels(models=models, features=features, channel_groups=channel_groups)
        confidence = _compute_confidence(postprob=postprob)
        sa, mm = _define_stage(confidence=confidence)

        for i, epoch in enumerate(self.epochs):
            epoch.stage = sa[i]
            epoch.max_confidence = mm[i]

        self.logger.info(msg="Finished scoring epochs.")
        return sa, mm

    def export_sleep_stage_output(self, output_folder: str, filename: str = 'SleepStage.csv') -> pd.DataFrame:
        """Exports sleep stage classifications to a CSV file.

        :param output_folder: Folder to save the output file.
        :param filename: Name of the output file. Defaults to 'SleepStage.csv'.

        :return: DataFrame containing the sleep stage classifications.
        """
        # Sort epochs per start time
        self.epochs.sort(key=lambda x:x.start_time)

        file_indentifiers = np.ones(shape=len(self.epochs))
        epoch_start_times_strings = [epoch.start_time.strftime('%Y-%m-%d %H:%M:%S') for epoch in self.epochs]
        stages = [epoch.stage for epoch in self.epochs]
        max_confidences = [round(epoch.max_confidence, 4) for epoch in self.epochs]
        epoch_start_samples = [epoch.matlab_start_sample for epoch in self.epochs]

        self.sleep_stages = pd.DataFrame({
            'FileIdentifier': file_indentifiers,
            'EpochStartTime': epoch_start_times_strings,
            'SleepStage': stages,
            'MaximumConfidence': max_confidences,
            'EpochStartSample': epoch_start_samples
        })

        self.sleep_stages.to_csv(os.path.join(output_folder, filename), index=False)

        return self.sleep_stages

    def export_summary_output(self, output_folder: str, filename: str = 'Summary.csv'):
        """Exports a summary of sleep stages and transitions to a CSV file.

        :param output_folder: Folder to save the output file.
        :param filename: Name of the output file. Defaults to 'Summary.csv'.

        :return: DataFrame containing the summary of sleep stages.
        """
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

            # dt = _matlab_datenum_to_datetime(Sl.loc[i, 'EpochStartTime'])
            dt_str = Sl.loc[i, 'EpochStartTime']
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
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

    def select_channels(self):
        """Opens a GUI for selecting channels to include in the analysis."""
        selected_channels = self.get_channel_selection()
        excluded_channels_names = set(self.channel_names) - set(selected_channels)
        self.excluded_channels = [Channel(name=chan_name) for chan_name in excluded_channels_names]
        return

    def get_channel_selection(self):
        """Creates a simple tkinter window for channel selection."""
        def submit():
            # Get selected indices and map to actual values
            selected_indices = listbox.curselection()
            selected_items = [options[i] for i in selected_indices]
            root.selected_options = selected_items
            root.destroy()

        def select_all():
            # Select all items in the listbox
            listbox.select_set(0, tk.END)

        def deselect_all():
            # Deselect all items
            listbox.selection_clear(0, tk.END)

        root = tk.Tk()
        root.title("Select Options")

        options = self.channel_names

        # Create a listbox with multiple selection mode
        listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
        for option in options:
            listbox.insert(tk.END, option)
        listbox.pack(padx=10, pady=10)

        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        # Select All button
        select_all_button = tk.Button(button_frame, text="Select All", command=select_all)
        select_all_button.pack(side=tk.LEFT, padx=5)

        # Deselect All button
        deselect_all_button = tk.Button(button_frame, text="Deselect All", command=deselect_all)
        deselect_all_button.pack(side=tk.LEFT, padx=5)

        # Submit button
        btn = tk.Button(root, text="OK", command=submit)
        btn.pack()

        root.selected_options = []
        root.mainloop()

        return root.selected_options


if __name__ == "__main__":
    pass