import datetime

import mne
import pyedflib
import numpy as np
import statistics

from ..layout import Channel, Montage


class EdfReader:
    """A class for reading and processing EDF files.

    This class provides functionality to read EDF files, extract metadata, and retrieve EEG data
    using either the `mne` or `pyedflib` libraries. It also includes methods for cleaning channel
    names, handling montages, and managing discarded channels.

    Attributes:
        filepath (str): Path to the EDF file.
        _raw_data (mne.io.Raw): Raw data object from the `mne` library.
        _channel_names (list): List of original channel names.
        channels (list): List of `Channel` objects representing cleaned channel names.
        _metadata (dict): Metadata extracted from the EDF file, including physical/digital ranges,
                          start time, and file duration.
        EPOCH_LENGTH (int): Length of an epoch in seconds (default: 30).

    Properties:
        channel_names (list): List of cleaned channel names.
        digital_max (np.ndarray): Maximum digital values for each channel.
        digital_min (np.ndarray): Minimum digital values for each channel.
        scaling_factor (np.ndarray): Scaling factor to convert digital values to physical units.
        signal_value_offset (np.ndarray): Offset to adjust signal baselines.
        start_datetime (datetime.datetime): Start time of the recording.
        start_time_sample (int): Sample index corresponding to the start time.
        start_timenum (float): Start time in MATLAB datenum format.
        physical_dimensions (dict): Physical units for each channel.
        physical_max (np.ndarray): Maximum physical values for each channel.
        physical_min (np.ndarray): Minimum physical values for each channel.
        n_chans (int): Number of channels in the recording.
        sampling_frequency (float): Sampling frequency of the recording.
        original_channel_names (list): Original channel names from the EDF file.
        file_duration (float): Total duration of the recording in seconds.
        data_offset (int): Byte offset where the data starts in the EDF file.
        n_records (int): Number of data records in the file.
        record_duration (float): Duration of a single record in seconds.
        samples_per_record (list): Number of samples per record for each channel.
        n_samples (int): Total number of samples in the recording.
        n_epochs (int): Total number of epochs in the recording.
        discarded_channels (list): List of channel indices with inconsistent samples per record.

    Methods:
        clean_channel_names(): Cleans and standardizes channel names.
        get_montage(): Returns a montage object for the channels.
        extract_data_mne(start_sample, end_sample, chan_picks=None): Extracts data using `mne`.
        extract_data_pyedflib(start_sample, stop_sample, chan_picks=None, digital=False): Extracts data using `pyedflib`.
        extract_data_raw(start_sample, end_sample): Extracts raw data directly from the EDF file.
    """

    _STELLATE_MIN_SECONDS_1 = 27.75
    _STELLATE_MIN_SECONDS_2 = 57.75

    _MATLAB_DATENUM_OFFSET = 719529
    _SECONDS_IN_DAY = 86400

    EPOCH_LENGTH = 30  # In seconds

    def __init__(self, filepath: str):
        self._raw_data = mne.io.read_raw_edf(input_fname=filepath, preload=False)
        self._channel_names = []
        self.channels = []
        self.filepath = filepath

        self._metadata = self._extract_metadata()

        assert int(self.sampling_frequency) == int(self.samples_per_record[0] / self.record_duration)
        assert int(self.n_samples) == int(self.n_records * self.samples_per_record[0])

    def _extract_metadata(self):
        metadata = dict()
        with pyedflib.EdfReader(self.filepath) as f:
            # TODO: Why are these values only 37 elements long?
            metadata['physical_max'] = f.getPhysicalMaximum()
            metadata['physical_min'] = f.getPhysicalMinimum()
            metadata['digital_min'] = f.getDigitalMinimum()
            metadata['digital_max'] = f.getDigitalMaximum()
            metadata['file_duration'] = f.getFileDuration()
            start_time = f.getStartdatetime()

        # TODO: check Stellate time compatibility thing, what is it?

        if start_time.second < self._STELLATE_MIN_SECONDS_1:
            start_time_sample = self.sampling_frequency * (29 - start_time.second)
            new_time_seconds = 29
        elif self._STELLATE_MIN_SECONDS_1 < start_time.second < self._STELLATE_MIN_SECONDS_2:
            start_time_sample = self.sampling_frequency * (59 - start_time.second)
            new_time_seconds = 59
        else:
            start_time_sample = self.sampling_frequency * (60 - start_time.second + 29)
            new_time_seconds = 29

        metadata['start_time'] = datetime.datetime(year=start_time.year, month=start_time.month, day=start_time.day,
                                                   hour=start_time.hour, minute=start_time.minute,
                                                   second=new_time_seconds)
        metadata['start_time_sample'] = round(start_time_sample)
        return metadata

    @property
    def channel_names(self):
        return [ch.name for ch in self.channels]

    @property
    def digital_max(self):
        """The integer upper bound of the range in which the data is stored, for each channel.
        Typically 32767 for 16-bit resolution."""
        return self._metadata['digital_max']

    @property
    def digital_min(self):
        """The integer lower bound of the range in which the data is stored, for each channel.
        Typically -32768 for 16-bit resolution."""
        return self._metadata['digital_min']

    @property
    def scaling_factor(self) -> np.ndarray:
        """Factor to scale stored digital range to real-world unit range, for each channel."""
        return np.divide((self.physical_max - self.physical_min), (self.digital_max - self.digital_min))

    @property
    def signal_value_offset(self) -> np.ndarray:
        """Constant added to shift signal to the correct baseline, for each channel."""
        return self.physical_min - np.multiply(self.digital_min, self.scaling_factor)

    @property
    def start_datetime(self):
        return self._metadata['start_time']

    @property
    def start_time_sample(self):
        return self._metadata['start_time_sample']

    @property
    def start_timenum(self):
        datenum = self._metadata['start_time'].timestamp() / self._SECONDS_IN_DAY + self._MATLAB_DATENUM_OFFSET
        return datenum

    @property
    def physical_dimensions(self):
        return self._raw_data._orig_units

    @property
    def physical_max(self):
        return self._metadata['physical_max']

    @property
    def physical_min(self):
        return self._metadata['physical_min']

    @property
    def n_chans(self):
        return self._raw_data.info['nchan']

    @property
    def sampling_frequency(self):
        return self._raw_data.info['sfreq']

    @property
    def original_channel_names(self):
        return self._raw_data.info['ch_names']

    @property
    def file_duration(self):
        return self._raw_data.duration

    @property
    def data_offset(self) -> int:
        """Length of the header, or starting offset for reading the data in the EDF file."""
        return self._raw_data._raw_extras[0]['data_offset']

    @property
    def n_records(self):
        return self._raw_data._raw_extras[0]['n_records']

    @property
    def record_duration(self):
        return self._raw_data._raw_extras[0]['record_length'][0]

    @property
    def samples_per_record(self):
        return self._raw_data._raw_extras[0]['n_samps']

    @property
    def n_samples(self):
        return self._raw_data.n_times

    @property
    def n_epochs(self):
        n_epochs = (self.n_samples - self.start_time_sample) / self.sampling_frequency / self.EPOCH_LENGTH

        # TODO: If n_epochs has decimals smaller than 0.042, we effectively go to the number of epochs lower integer, why?
        if n_epochs % 1 < 1.25 / self.EPOCH_LENGTH:
            n_epochs -= 0.05

        return np.floor(n_epochs).astype(int)

    def clean_channel_names(self):
        original_chans = self._raw_data.info['ch_names']

        clean_channels = []
        for ch in original_chans:
            channel = Channel(original_name=ch)
            channel.clean_name()
            clean_channels.append(channel)

        self.channels = clean_channels

    def get_montage(self):
        return Montage(channels=self.channels, montage_type='referential')

    def extract_data_mne(self, start_sample: int, end_sample: int, chan_picks: list = None):
        """
        Reads the data between start and stop samples for the specified channels.
        Data should be automatically scaled to be converted to physical units. Make sure that's the case.
        MNE's method to extract data should also account for where the data actually starts, so we shouldn't need to
        add the data_offset to the start and stop samples.

        Note that MNE's get_data() excludes the stop sample. We're changing it such that stop sample is included in the
        window to extract.

        :param start_sample: Sample from which to start extracting the data.
        :param end_sample: Last sample to extract from the data.
        :param chan_picks: List of channel names for which to read the data. Default is None to read all EEG channels.
        :return: Numpy array containing the data. (n_chans x n_samples)
        """
        # TODO: gotta assert that the number of channels is same as the length of samples per record.
        # TODO: not the case as yet
        # Make sure self.channels is not empty
        if not self.channels:
            self.clean_channel_names()

        if chan_picks is None:
            chan_picks = [ch.original_name for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
            channels = [ch for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
        else:
            channels = [ch for i, ch in enumerate(self.channels) if ch.name in chan_picks]

        data, times = self._raw_data.get_data(picks=chan_picks, start=start_sample, stop=end_sample, return_times=True)

        return data*1e6, times, channels

    def extract_data_pyedflib(self, start_sample: int, stop_sample: int, chan_picks: list = None, digital: bool=False):
        """
        Reads the data between start and stop samples for the specified channels.
        Data is automatically scaled to be converted to physical units.
        Pyedflib's method to extract data should also account for where the data actually starts, so we shouldn't need to
        add the data_offset to the start and stop samples.

        Note that MNE's get_data() excludes the stop sample. We're changing it such that stop sample is included in the
        window to extract.

        :param start_sample: Sample from which to start extracting the data.
        :param stop_sample: Last sample to extract from the data.
        :param chan_picks: List of channel names for which to read the data. Default is None to read all EEG channels.
        :return: Numpy array containing the data. (n_chans x n_samples)
        """
        # TODO: gotta assert that the number of channels is same as the length of samples per record.
        # TODO: not the case as yet
        # Make sure self.channels is not empty
        if not self.channels:
            self.clean_channel_names()

        if chan_picks is None:
            channels = [ch for i, ch in enumerate(self.channels) if i not in self.discarded_channels]
        else:
            channels = [ch for i, ch in enumerate(self.channels) if ch.name in chan_picks]

        edf = pyedflib.EdfReader(file_name=self.filepath)

        data_list = []
        for i in range(len(channels)):
            data_list.append(edf.readSignal(chn=i, start=start_sample, n=(stop_sample-start_sample), digital=digital))

        units = edf.getPhysicalDimension(chn=0)

        if digital:
            data = np.array(data_list).astype('int16')
        else:
            data = np.array(data_list)

        return data, channels

    def extract_data_raw(self, start_sample, end_sample):
        with open(self.filepath, "rb") as f:
            # Read the fixed 256-byte header
            header = f.read(256).decode('ascii', errors='ignore')

            # Extract metadata
            num_signals = int(header[252:256].strip())  # Number of signals (channels)
            recording_duration = float(header[244:252].strip())  # Total duration in seconds
            num_records = int(header[236:244].strip())  # Number of data records
            record_duration = float(header[232:236].strip()) if  header[232:236].strip() != '' else np.nan # Duration of one record in seconds

            f.seek(256)  # Move to where channel info starts

            # Read per-channel metadata (16 bytes per channel for 8 fields = 256 bytes per channel)
            channel_labels = []
            sample_rates = []
            sensors = []
            units = []
            physmin = []
            physmax = []
            digmin = []
            digmax = []
            preproc = []
            num_samples_per_record = []

            for _ in range(num_signals):
                channel_labels.append(f.read(16).decode('ascii').strip())  # Channel name
            for _ in range(num_signals):
                sensors.append(f.read(80).decode('ascii').strip())
            for _ in range(num_signals):
                units.append(f.read(8).decode('ascii').strip())
            for _ in range(num_signals):
                physmin.append(f.read(8).decode('ascii').strip())
            for _ in range(num_signals):
                physmax.append(f.read(8).decode('ascii').strip())
            for _ in range(num_signals):
                digmin.append(f.read(8).decode('ascii').strip())
            for _ in range(num_signals):
                digmax.append(f.read(8).decode('ascii').strip())
            for _ in range(num_signals):
                preproc.append(f.read(80).decode('ascii').strip())
            for _ in range(num_signals):
                num_samples_per_record.append(int(f.read(8).decode('ascii').strip()))

            # Compute sampling frequency (assuming all channels have the same rate)
            fs = num_samples_per_record[0] / record_duration

            # Compute the byte offset to seek directly to the desired range
            num_samples_total = sum(num_samples_per_record)
            bytes_per_sample = 2  # int16 (2 bytes per sample)
            len = 256 + num_signals * 256  # Position where data begins

            # Seek to the start of the time window
            f.seek(len + start_sample * num_signals * bytes_per_sample)

            # Read only the required samples
            num_samples_to_read = end_sample - start_sample
            raw_data = np.fromfile(f, dtype='<i2', count=num_samples_to_read * num_signals)

            # Reshape
            raw_data = raw_data.reshape((num_signals, num_samples_to_read))

            return raw_data.T, channel_labels

    @property
    def discarded_channels(self):
        """ Disregard channels that have different number of samples per record than the others

        :return:
        """

        # TODO: What to do when samples per record is bigger than the number of channels, like in the case of a channel of annotations?

        main_mode = statistics.mode(self.samples_per_record)
        discarded_chans = np.where(self.samples_per_record != main_mode)[0]
        return discarded_chans if len(discarded_chans) > 0 else None