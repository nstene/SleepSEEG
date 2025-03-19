from models.readers.edf_reader import EdfReader

class EEGReaderFactory:
    """Factory to create the correct EEG loader based on file format."""

    @staticmethod
    def get_reader(file_path: str):
        # TODO ake it case insensitive
        if file_path.endswith(".edf"):
            return EdfReader(file_path)
        else:
            raise ValueError("Unsupported EEG file format.")
