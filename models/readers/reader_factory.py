from models.readers.edf_reader import EdfReader
from models.readers.mef_reader import MefReader


class EEGReaderFactory:
    """Factory to create the correct EEG loader based on file format."""

    @staticmethod
    def get_reader(file_path: str):
        # TODO ake it case insensitive
        if file_path.endswith(".edf"):
            return EdfReader(filepath=file_path)
        elif file_path.endswith(".mefd"):
            return MefReader(filepath=file_path)
        else:
            raise ValueError("Unsupported EEG file format.")
