import typing as t
import warnings

import numpy as np

class Channel:
    """Represents a channel with properties for name, family, and bipolar form.

    Attributes:
        original_name (str): The original name of the channel, as provided.
        family_index (int or float): The index of the channel within its family.
            Set to `np.nan` if no index is found.
        chan_family (str): The family or group to which the channel belongs.
        name (str): The cleaned and standardized name of the channel.

    Methods:
        bipolar: Returns the bipolar form of the channel as a list of two electrode names.
        clean_name: Standardizes the original channel name and extracts family and index.
    """
    def __init__(self, original_name: str=None, name: str=None, chan_family: str=None):
        """Initializes the Channel object.

        Args:
            original_name (str, optional): The original name of the channel. Defaults to None.
            name (str, optional): The cleaned name of the channel. Defaults to None.
            chan_family (str, optional): The family or group to which the channel belongs.
                Defaults to None.
        """
        self.original_name = original_name
        self.family_index = None
        self.chan_family = chan_family
        self.name = name

    @property
    def bipolar(self) -> t.List[str]:
        """Returns the bipolar form of the channel as a list of two electrode names.
        The channel name is expected to be in the format 'elec_name_1-elec_name_2'.
        """
        splitted_name = self.name.split('-')
        bipolar_chan = []
        if len(splitted_name) == 2:
            bipolar_chan.append(splitted_name[0])
            bipolar_chan.append(splitted_name[1])
        return bipolar_chan

    def clean_name(self) -> None:
        """Standardizes the original channel name.

        Ex:
        self.original_channel_name = 'EEG LTP1     '

        will result in

        self.chan_family = 'LTP'
        self.name = 'LTP1'
        self.family_index = 1
        """
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
    """Represents a montage of channels, supporting referential and bipolar configurations.

    Attributes:
        MONTAGE_TYPES (list[str]): List of valid montage types: 'referential' and 'bipolar'.
        channels (list[Channel]): List of channels in the montage.
        montage_type (str): Type of montage, either 'referential' or 'bipolar'.

    Methods:
        channel_families: Returns a dictionary of channel families and their associated channels.
        channel_names: Returns a list of channel names in the montage.
        make_bipolar_montage: Converts a referential montage to a bipolar montage.
    """

    MONTAGE_TYPES = ['referential', 'bipolar']

    def __init__(self, channels: t.List[Channel], montage_type: str):
        """Initializes the Montage object.

        Args:
            channels (list[Channel]): List of Channel objects to include in the montage.
            montage_type (str): Type of montage, either 'referential' or 'bipolar'.
        """
        if montage_type not in self.MONTAGE_TYPES:
            raise ValueError(f'Invalid montage_type {montage_type}. Please choose between "referential" and "bipolar"')

        self.channels = channels
        self.montage_type = montage_type

    @property
    def channel_families(self) -> t.Dict[str, t.List[Channel]]:
        """Returns a dictionary of channel families and their associated channels."""
        channel_families = dict()
        for ch in self.channels:
            if ch.chan_family not in channel_families:
                channel_families[ch.chan_family] = []
            channel_families[ch.chan_family].append(ch)

        return channel_families

    @property
    def channel_names(self) -> t.List[str]:
        """Returns the channel names in the montage."""
        return [ch.name for ch in self.channels]

    def make_bipolar_montage(self) -> None:
        """Converts a referential montage to a bipolar montage.

        Ex:
        If the montage contains channels ['LTP1', 'LTP2', 'LTP3'],
        this method will create bipolar channels ['LTP1-LTP2', 'LTP2-LTP3'].
        """

        if self.montage_type == 'bipolar':
            warnings.warn("Montage is already bipolar")
            return

        chan_families = self.channel_families.copy()

        # Reinitialize the montage properties
        self.montage_type = None
        self.channels = None

        bipolar_channels = []
        for chan_fam, members in chan_families.items():
            # order members by their family index
            members_sorted = sorted(members, key=lambda obj: obj.family_index)
            # if consecutive, make channel pair
            for i in range(len(members_sorted) - 1):
                if members_sorted[i].family_index == members_sorted[i + 1].family_index - 1:
                    new_bipolar_chan_name = members_sorted[i].name + '-' + members_sorted[i + 1].name
                    bipolar_channels.append(Channel(name=new_bipolar_chan_name, chan_family=chan_fam))

        self.montage_type = 'bipolar'
        self.channels = bipolar_channels


class EEG:
    """Represents an EEG recording with associated data, montage, and timestamps.

    Attributes:
        montage (Montage, optional): The montage associated with the EEG recording. Defaults to None.
        _timestamps (np.ndarray, optional): Timestamps for the EEG data. Defaults to None.
        _chan_idx (dict[str, int]): A dictionary mapping channel names to their indices in the data array.
        _data (np.ndarray): The EEG data array with shape (n_channels, n_samples).

    Methods:
        chan_idx: Returns a dictionary containing the index of each channel in the montage.
        set_chan_idx: Sets the channel indices dictionary.
        make_bipolar_montage: Converts the EEG recording to a bipolar montage.
    """
    def __init__(self, data: np.ndarray, montage: Montage=None, timestamps: np.ndarray=None):
        """Initializes the EEG object.

        Args:
            data (np.ndarray): The EEG data array with shape (n_channels, n_samples).
            montage (Montage, optional): The montage associated with the EEG recording. Defaults to None.
            timestamps (np.ndarray, optional): Timestamps for the EEG data. Defaults to None.
        """
        self.montage = montage
        self._timestamps = timestamps

        self._chan_idx = {}
        self._data = data

        if montage:
            assert data.shape[0] == len(montage.channel_names)
            for i, ch in enumerate(montage.channels):
                self._chan_idx[ch.name] = i

    @property
    def chan_idx(self) -> t.Dict[str, int]:
        """Returns a dictionary containing the index of each channel in the montage."""
        return self._chan_idx

    def set_chan_idx(self, chan_idx: t.Dict[str, int]) -> None:
        """Sets the channel indices dictionary.

        :param chan_idx: Dicitonary containing the new indices for each channel
        """
        self._chan_idx = chan_idx

    def make_bipolar_montage(self) -> None:
        """Converts the EEG recording to a bipolar montage.

        This method modifies the EEG data and montage to represent a bipolar configuration. It computes the difference
        between consecutive channels in the referential montage to create bipolar pairs.
        The original data is overwritten with the new bipolar data.
        """
        if self.montage is None:
            raise AttributeError("Specify a referential montage before creating a bipolar montage.")

        self.montage.make_bipolar_montage()

        new_signals_list = []
        for chan_pair in self.montage.channels:
            chan_1_name = chan_pair.bipolar[0]
            chan_2_name = chan_pair.bipolar[1]

            chan_1_idx = self._chan_idx[chan_1_name]
            chan_2_idx = self._chan_idx[chan_2_name]

            new_signals_list.append(self._data[chan_1_idx] - self._data[chan_2_idx])

        bipolar_data = np.vstack(new_signals_list)

        data_list = []
        self._chan_idx = {}
        for i, ch in enumerate(self.montage.channels):
            data_list.append(bipolar_data[i, :])
            self._chan_idx[ch.name] = i
        self._data = np.vstack(data_list)


if __name__ == "__main__":
    pass