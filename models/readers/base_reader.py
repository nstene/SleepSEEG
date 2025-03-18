import abc

class BaseEEGLoader(abc.ABC):
    """Abstract base class for EEG readers."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw = None

    @abc.abstractmethod
    def load_data(self, start_sample, end_sample, chan_picks):
        """Loads EEG data. Must be implemented by subclasses."""
        pass