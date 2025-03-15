import abc

class BaseEEGLoader(abc.ABC):
    """Abstract base class for EEG readers."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw = None

    @abc.abstractmethod
    def load_data(self):
        """Loads EEG data. Must be implemented by subclasses."""
        pass