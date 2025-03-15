import datetime

import mne
from pymef.mef_session import MefSession
import pyedflib
from matplotlib import pyplot as plt

import numpy as np
import statistics
import tkinter as tk

import typing as t
import warnings

# TODO: I want EEG Reader to end up creating epochs


class Channel:
    def __init__(self, original_name=None, name=None, chan_family=None):
        self.original_name = original_name
        self.family_index = None
        self.chan_family = chan_family
        self.name = name

    @property
    def bipolar(self):
        splitted_name = self.name.split('-')
        bipolar_chan = []
        if len(splitted_name) == 2:
            bipolar_chan.append(splitted_name[0])
            bipolar_chan.append(splitted_name[1])
        return bipolar_chan

    def clean_name(self):
        chan_stripped = self.original_name.strip()  # Strip trailing whitespaces
        chan_name = " ".join(chan_stripped.split()[1:])  # Split after first whitespace, getting rid of electrode type

        if chan_name != '':
            clean_name = chan_name
        else:
            clean_name = chan_stripped

        if not chan_stripped[-1].isdigit():
            chan_family = clean_name
            family_index = np.nan
        elif not chan_stripped[-2].isdigit():
            chan_family = clean_name[:-1]
            family_index = int(clean_name[-1:])
        else:
            chan_family = clean_name[:-2]
            family_index = int(clean_name[-2:])

        self.chan_family = chan_family
        self.name = clean_name
        self.family_index = family_index

        return


class Montage:
    def __init__(self, channels: t.List[Channel], montage_type: str):

        MONTAGE_TYPES = ['referential', 'bipolar']
        if montage_type not in MONTAGE_TYPES:
            raise ValueError(f'Invalid montage_type {montage_type}. Please choose between "referential" and "bipolar"')

        self.channel_families = None
        self.channels = channels
        self.montage_type = montage_type
        self._extract_channel_families()

    @property
    def channel_names(self):
        return [ch.name for ch in self.channels]

    def _extract_channel_families(self):
        channel_families = dict()
        for ch in self.channels:
            if ch.chan_family not in channel_families:
                channel_families[ch.chan_family] = []
            channel_families[ch.chan_family].append(ch)

        self.channel_families = channel_families

    def make_bipolar_montage(self):

        if self.montage_type == 'bipolar':
            warnings.warn("Montage is already bipolar")
            return

        # Reinitialize the montage properties
        self.montage_type = None
        self.channels = None

        bipolar_channels = []
        for chan_fam, members in self.channel_families.items():
            # order members by their family index
            members_sorted = sorted(members, key=lambda obj: obj.family_index)
            # if consecutive, make channel pair
            for i in range(len(members_sorted) - 1):
                if members_sorted[i].family_index == members_sorted[i + 1].family_index - 1:
                    new_bipolar_chan_name = members_sorted[i].name + '-' + members_sorted[i + 1].name
                    bipolar_channels.append(Channel(name=new_bipolar_chan_name, chan_family=chan_fam))

        self.montage_type = 'bipolar'
        self.channels = bipolar_channels

    def select_channels_gui(self):
        def get_selection(options):
            """ Creates a simple tkinter window with checkboxes for multiple selection. """
            selected_options = []

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

        return get_selection(options=self.channel_names)


class EEG:
    def __init__(self, data: np.ndarray, montage: Montage = None, timestamps= None):
        self._montage = montage
        self._timestamps = timestamps

        self._chan_idx = {}
        self._data = data

        if montage:
            assert data.shape[0] == len(montage.channel_names)
            for i, ch in enumerate(montage.channels):
                self._chan_idx[ch.name] = i

    def make_bipolar_montage(self):
        self._montage.make_bipolar_montage()

        new_signals_list = []
        for chan_pair in self._montage.channels:
            chan_1_name = chan_pair.bipolar[0]
            chan_2_name = chan_pair.bipolar[1]

            chan_1_idx = self._chan_idx[chan_1_name]
            chan_2_idx = self._chan_idx[chan_2_name]

            new_signals_list.append(self._data[chan_1_idx] - self._data[chan_2_idx])

        bipolar_data = np.vstack(new_signals_list)

        data_list = []
        self._chan_idx = {}
        for i, ch in enumerate(self._montage.channels):
            data_list.append(bipolar_data[i, :])
            self._chan_idx[ch.name] = i
        self._data = np.vstack(data_list)


class EdfReader:
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


if __name__ == "__main__":
    filepath = r'../eeg_data/auditory_stimulation_P18_002.edf'
    edf = EdfReader(filepath)
    edf.clean_channel_names()

    window_center = edf.start_time_sample
    window_length_seconds = 2.5
    window_length_samples = 2.5 * edf.sampling_frequency
    start_sample = window_center - round(window_length_samples / 2) + 1
    #start_sample = 0
    stop_sample = window_center + round(window_length_samples / 2 + 30 * edf.sampling_frequency)

    # TODO: Figure out units: pyedflib returns physical signal in microvolts and mne in volts (10-6 ratio between both).
    # Matlab in microvolts

    eeg_mne, times, chans = edf.extract_data_mne(start_sample=start_sample, end_sample=stop_sample)
    eeg_pyedflib = edf.extract_data_pyedflib(start_sample=start_sample, stop_sample=stop_sample)
    #eeg_pyedflib_dig = edf.extract_data_pyedflib(start_sample=start_sample, stop_sample=stop_sample, digital=True)
    #eeg_raw, chan_labels = edf.extract_data_raw(start_sample=start_sample, end_sample=stop_sample+1)

    scaling_factor = edf.scaling_factor
    offset = edf.signal_value_offset
    offset[scaling_factor < 0] = 0
    scaling_factor[scaling_factor < 0] = 1

    # eeg_pyedflib.make_bipolar_montage()

    plt.plot(eeg_pyedflib._data[0])
    plt.show()

    # TODO: See why data not same as matlab's

    scaling_factor = edf.scaling_factor
    offset = edf.signal_value_offset
    offset[scaling_factor < 0] = 0
    scaling_factor[scaling_factor < 0] = 1


    #selected_channels = montage.select_channels_gui()

    # TODO: Extract epochs from file
    # TODO: Careful with channel units: get these from the file info, some are microvolts as expected, some are millivolts

    #print(selected_channels)
