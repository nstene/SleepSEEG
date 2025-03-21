import typing as t

from models.readers.edf_reader import EdfReader
from models.readers.med_reader import MedReader
from models.readers.mef_reader import MefReader


class EEGReaderFactory:
    """Factory to create the correct EEG loader based on file format."""

    @staticmethod
    def get_reader(file_path: str, password: t.Optional[str]):
        # TODO ake it case insensitive
        if file_path.endswith(".edf"):
            return EdfReader(filepath=file_path)
        elif file_path.endswith(".mefd"):
            return MefReader(filepath=file_path)
        elif file_path.endswith(".medd"):
            return MedReader(filepath=file_path, password=password)
        else:
            raise ValueError("Unsupported EEG file format.")
