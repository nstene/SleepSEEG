import abc
import datetime
import typing as t


class BaseEEGReader(abc.ABC):
    """Abstract base class for EEG readers."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw = None

    @property
    @abc.abstractmethod
    def channel_names(self) -> t.List[str]:
        """Returns channel names. Must be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def start_time_sample(self) -> int:
        """Returns the sample at which to begin reading the data. Must be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def start_datetime(self) -> datetime.datetime:
        """Returns the start date and time of the recording. Must be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def sampling_frequency(self) -> int:
        """Returns the sampling frequency of the recording. Must be implemented by subclasses."""
        pass

    @abc.abstractmethod
    def load_data(self, start_sample, end_sample, chan_picks):
        """Loads EEG data. Must be implemented by subclasses."""
        pass