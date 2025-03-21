from models.readers.base_reader import BaseEEGReader
from pymef.mef_session import MefSession
from pymef.mef_file import pymef3_file
# from mef_tools.io import MefReader
import typing as t
import numpy as np
import statistics
from models.layout import Channel, Montage

from datetime import datetime

# TODO: Make sure the the digital to physical conversion is done right.

class MefReader(BaseEEGReader):
    """A class for reading, processing, and extracting data from MEF files.

    This class provides functionality to read MEF files, extract metadata, and retrieve EEG data.
    """

    _STELLATE_MIN_SECONDS_1 = 27.75
    _STELLATE_MIN_SECONDS_2 = 57.75

    EPOCH_LENGTH = 30  # In seconds

    def __init__(self, filepath: str):
        """Initializes the MefReader object.

        Args:
            filepath (str): Path to the EDF file to be read.
        """

        super().__init__(filepath)
        self.montage = None
        password = ''

        self._ms = MefSession(filepath, password)
        self._info = self._ms.read_ts_channel_basic_info()

        self._metadata = self._extract_metadata()

        self.channels = [Channel(original_name=chan) for chan in self.original_channel_names]

    def get_montage(self) -> Montage:
        """Returns a referential montage instance populated with the channels in the recoring."""
        self.montage = Montage(channels=self.channels, montage_type='referential')
        return self.montage

    def _extract_metadata(self):
        metadata = dict()

        # Sampling frequency
        # TODO: Make it more robust
        fs = [chan['fsamp'] for chan in self._info][0][0]

        # Start time
        start_times_microseconds = [chan['start_time'][0] for chan in self._info]
        recording_start_time_seconds = start_times_microseconds[0] // 1_000_000
        recording_start_time_microseconds = start_times_microseconds[0] % 1_000_000
        start_time = datetime.fromtimestamp(recording_start_time_seconds).replace(
            microsecond=recording_start_time_microseconds).time()
        if start_time.second < self._STELLATE_MIN_SECONDS_1:
            start_time_sample = fs * (29 - start_time.second)
            new_time_seconds = 29
        elif self._STELLATE_MIN_SECONDS_1 < start_time.second < self._STELLATE_MIN_SECONDS_2:
            start_time_sample = fs * (59 - start_time.second)
            new_time_seconds = 59
        else:
            start_time_sample = fs * (60 - start_time.second + 29)
            new_time_seconds = 29

        metadata['start_time'] = datetime(year=1970, month=1, day=1,
                                                   hour=start_time.hour, minute=start_time.minute,
                                                   second=new_time_seconds)
        metadata['start_time_sample'] = round(start_time_sample)
        return metadata

    @property
    def channel_names(self) -> t.List[str]:
        return [chan.name for chan in self.channels]

    @property
    def start_time_sample(self) -> int:
        """Returns the sample at which to begin reading the data."""
        return self._metadata['start_time_sample']

    @property
    def start_datetime(self) -> datetime:
        """Returns the start date and time of the recording."""
        return self._metadata["start_time"]

    @property
    def n_samples(self):
        return [chan['nsamp'] for chan in self._info][0][0]

    @property
    def sampling_frequency(self):
        return [chan['fsamp'] for chan in self._info][0][0]

    @property
    def n_epochs(self) -> int:
        """Returns the number of 30s epochs that can be extracted from the recording."""
        n_epochs = (self.n_samples - self.start_time_sample) / self.sampling_frequency / self.EPOCH_LENGTH

        # TODO: If n_epochs has decimals smaller than 0.042, we effectively go to the number of epochs lower integer, why?
        if n_epochs % 1 < 1.25 / self.EPOCH_LENGTH:
            n_epochs -= 0.05

        return np.floor(n_epochs).astype(int)

    @property
    def original_channel_names(self) -> t.List[str]:
        """Returns the channel names as they were originally extracted from the recording."""
        return [chan['name'] for chan in self._info]

    def load_data(self, start_sample: int, end_sample: int, chan_picks: list = None) \
            -> t.Tuple[np.ndarray[float], np.ndarray[float]|None, t.List[Channel]]:

        if chan_picks is None:
            chan_picks = [ch.original_name for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
            channels = [ch for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
        else:
            channels = [ch for i, ch in enumerate(self.channels) if ch.name in chan_picks]

        data = np.array(self._ms.read_ts_channels_sample(chan_picks, [[start_sample, end_sample]]))

        ufactor = np.array([self._info[i]['ufact'][0] for i in range(len(self.channels)) if i not in self.discarded_channels])

        return np.multiply(data, np.tile(ufactor[:, np.newaxis], data.shape[1])), None, channels

    @property
    def discarded_channels(self) -> np.ndarray[int]|None:
        """Disregard channels that have different number of samples per record than the others.

        :return: Numpy array containing the indices of the channels to exclude, if any, or None.
        """

        # TODO: What to do when samples per record is bigger than the number of channels, like in the case of a channel of annotations?

        n_samples_channels = [chan['nsamp'][0] for chan in self._info]
        main_mode = statistics.mode(n_samples_channels)
        discarded_chans = np.where(n_samples_channels != main_mode)[0]
        return discarded_chans if len(discarded_chans) > 0 else []


if __name__=="__main__":
    filepath = r'C:\Users\natha\Documents\projects\SleepSEEG\eeg_data\ds003708\ds003708\sub-01\ses-ieeg01\ieeg\sub-01_ses-ieeg01_task-ccep_run-01_ieeg.mefd'
    mef_reader = MefReader(filepath=filepath)