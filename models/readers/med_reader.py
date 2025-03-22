import warnings
from datetime import datetime
import typing as t
import re

import numpy as np
import statistics
from dhn_med_py import MedSession

from ..layout import Channel, Montage
from models.readers.base_reader import BaseEEGReader

# TODO handle sampling frequency per channel

class MedReader(BaseEEGReader):

    _STELLATE_MIN_SECONDS_1 = 27.75
    _STELLATE_MIN_SECONDS_2 = 57.75
    EPOCH_LENGTH = 30  # In seconds

    def __init__(self, filepath: str, password: str):
        """Initializes the EdfReader object.

        Args:
            filepath (str): Path to the EDF file to be read.
        """
        super().__init__(filepath=filepath)
        # open session
        self._sess = MedSession(session_path=filepath, password=password)
        self._info = self._sess.session_info
        self._reference_channel_idx = [i for i, chan in enumerate(self._info['channels']) if chan['metadata']['channel_name'] == self._sess.reference_channel][0]
        self.channels = [Channel(original_name=chan) for chan in self.original_channel_names]

        self._metadata = self._extract_metadata()
        self.montage = None

        # Make some checks
        sampling_frequencies = [chan['metadata']['sampling_frequency'] for chan in self._info['channels']]
        if not np.all(sampling_frequencies == sampling_frequencies[0]):
            warnings.warn("Channels with different sampling frequencies are not yet supported.\n "
                          "The reference channel's sampling frequency will be used throughout the analysis, which give invalid results.")

    def _extract_metadata(self):
        metadata = {}
        # Start time
        start_date_string = self._info['metadata']['start_time_string']
        date_format = "%a %d %b %Y %H:%M:%S.%f"
        datetime_part = re.sub(r"\s[A-Z]{2,4}\s\(UTC.*\)", "", start_date_string)
        start_datetime = datetime.strptime(datetime_part, date_format)

        # End time
        end_date_string = self._info['metadata']['end_time_string']
        datetime_part = re.sub(r"\s[A-Z]{2,4}\s\(UTC.*\)", "", end_date_string)
        end_datetime = datetime.strptime(datetime_part, date_format)
        metadata['end_time'] = end_datetime

        if start_datetime.second < self._STELLATE_MIN_SECONDS_1:
            start_time_sample = self.sampling_frequency * (29 - start_datetime.second)
            new_time_seconds = 29
        elif self._STELLATE_MIN_SECONDS_1 < start_datetime.second < self._STELLATE_MIN_SECONDS_2:
            start_time_sample = self.sampling_frequency * (59 - start_datetime.second)
            new_time_seconds = 59
        else:
            start_time_sample = self.sampling_frequency * (60 - start_datetime.second + 29)
            new_time_seconds = 29

        metadata['start_time'] = datetime(year=start_datetime.year, month=start_datetime.month, day=start_datetime.day,
                                          hour=start_datetime.hour, minute=start_datetime.minute,
                                          second=new_time_seconds)
        metadata['start_time_sample'] = round(start_time_sample)

        return metadata

    def load_data(self, start_sample: int, end_sample: int, chan_picks: list = None) \
            -> t.Tuple[np.ndarray, None, t.List[Channel]]:
        # TODO: this is not working

        if chan_picks is None:
            chan_picks = [ch.original_name for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
            channels = [ch for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
        else:
            channels = [ch for i, ch in enumerate(self.channels) if ch.name in chan_picks]

        med_dat_matrix = self._sess.data_matrix.get_matrix_by_index(start_sample, end_sample, sampling_frequency=500)

        data = self._sess.read_by_index(0, 1, channels=chan_picks[0])
        return data, None, channels


    @property
    def sampling_frequency(self) -> int:
        """Returns sampling frequency.
        It is assumed all channels have the same sampling frequency, so we take the reference channel's sampling frequency.
        """
        return self._info['channels'][self._reference_channel_idx]['metadata']['sampling_frequency']

    @property
    def channel_names(self) -> t.List[str]:
        """Returns a list of clean channel names."""
        return [chan.name for chan in self.channels]

    @property
    def start_datetime(self) -> datetime:
        """Returns the start date and time of the recording."""
        return self._metadata['start_time']

    @property
    def end_datetime(self) -> datetime:
        """Returns the end date and time of the recording."""
        return self._metadata['end_time']

    @property
    def start_time_sample(self) -> int:
        """Returns the sample at which to begin reading the data."""
        return self._metadata['start_time_sample']

    @property
    def original_channel_names(self):
        """Returns the channel names as they were originally extracted from the recording."""
        return [chan['metadata']['channel_name'] for chan in self._info['channels']]

    def get_montage(self) -> Montage:
        """Returns a referential montage instance populated with the channels in the recoring."""
        self.montage = Montage(channels=self.channels, montage_type='referential')
        return self.montage

    @property
    def n_samples(self):
        """Returns the total number of samples of the recording.
        It is assumed all channels have the same sampling frequency, so we take the reference channel's fs.
        """
        diff = self.end_datetime - self.start_datetime
        return diff.seconds * self.sampling_frequency

    @property
    def n_epochs(self):
        """Returns the number of 30s epochs that can be extracted from the recording."""
        n_epochs = (self.n_samples - self.start_time_sample) / self.sampling_frequency / self.EPOCH_LENGTH

        # TODO: If n_epochs has decimals smaller than 0.042, we effectively go to the number of epochs lower integer, why?
        if n_epochs % 1 < 1.25 / self.EPOCH_LENGTH:
            n_epochs -= 0.05

        return np.floor(n_epochs).astype(int)

    @property
    def discarded_channels(self) -> np.ndarray[int] | None:
        """Disregard channels that have different number of samples per record than the others.

        :return: Numpy array containing the indices of the channels to exclude, if any, or None.
        """

        # TODO: What to do when samples per record is bigger than the number of channels, like in the case of a channel of annotations?

        # n_samples_channels = [chan['nsamp'][0] for chan in self._info]
        # main_mode = statistics.mode(n_samples_channels)
        # discarded_chans = np.where(n_samples_channels != main_mode)[0]
        # return discarded_chans if len(discarded_chans) > 0 else []
        return []
