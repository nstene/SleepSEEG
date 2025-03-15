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


if __name__ == "__main__":
    pass