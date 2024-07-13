import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from IPython.display import display, clear_output

import pkgutil
import spiketag
from spiketag.realtime import BMI, Binner
from spiketag.analysis import decoder
from spiketag.base import probe

import serial
from scipy.ndimage import gaussian_filter1d

class BMIRealtime:
    def __init__(self, prb_path, fetfile, ttlport):
        prb = probe()
        prb.load(prb_path)
        self.bmi = BMI(prb=prb, fetfile=fetfile, ttlport=ttlport)
        self.spike_times_dict = {}
        self.colors = plt.cm.get_cmap('tab10', 10)  # change the number if you need
        self.last_signal_time = 0
        self.laser_duration = 1 # sec

    def plot_raster(self, bmi_output, window_duration, updateFlag):
        t = bmi_output.timestamp  # time
        spk_id = bmi_output.spk_id
        rm_t = t % window_duration

        if spk_id not in self.spike_times_dict:
            self.spike_times_dict[spk_id] = []

        if updateFlag == 1:
            plt.close()
            for neuron_id in self.spike_times_dict:
                self.spike_times_dict[neuron_id].clear()

        self.spike_times_dict[spk_id].append(rm_t)

        clear_output(wait=True)
        plt.figure(figsize=(10, 8))

        all_spike_times = [self.spike_times_dict[neuron_id] for neuron_id in sorted(self.spike_times_dict.keys())]
        color_list = [self.colors(i % 10) for i in range(len(all_spike_times))]

        plt.eventplot(all_spike_times, orientation='horizontal', linelengths=0.8, colors=color_list)
        plt.xlim(0, window_duration)
        plt.xlabel('Time (ms)')
        plt.title('Real-time Raster Plot for Neuron units')
        plt.show()
#         time.sleep(0.1)
        
        
    def send_signal_to_teensy(self):
        if self.bmi.TTLserial is not None:
            self.bmi.TTLserial.write(b'a')
            self.bmi.TTLserial.flush()
            self.last_signal_time = time.time()

    def reset_signal_to_teensy(self):
        if self.bmi.TTLserial is not None:
            current_time = time.time()
            if current_time - self.last_signal_time >= self.laser_duration:
                self.bmi.TTLserial.write(b'b')
                self.bmi.TTLserial.flush()
    
    def bmi_func(self, mode, targetID, neuron_id, threshold, window_duration):
        start_time = 0
        while True:
            updateFlag = 0
            frFlag = 0
            spkFlag = 0
            bmi_output = self.bmi.read_bmi()
            # for plot_raster
            if bmi_output.timestamp - start_time > window_duration:
                updateFlag = 1
                start_time = bmi_output.timestamp
            
            self.plot_raster(bmi_output, window_duration,updateFlag)
            
            if mode != "spike ID":
                self.bmi.binner.input(bmi_output)
                count_vec = self.bmi.binner.output
                sum_count = count_vec.sum(axis=0)
                if sum_count[neuron_id] >= threshold:
                    frFlag = 1
            if mode != "firing rate":
                spike_id = bmi_output.spk_id
                if spike_id == targetID:
                    spkFlag = 1

            if frFlag == 1 or spkFlag == 1:
                self.send_signal_to_teensy()
            else:
                self.reset_signal_to_teensy()

    def start_bmi_realtime(self, bsize, Bbins, neuron_id, threshold, t_smooth,
                           bmi_update_rule, posterior_threshold, two_steps_decoding, mode, targetID, window_duration, laser_duration):
        pos_buffer_len = int(float(t_smooth) / float(bsize))
        self.bmi.bmi_update_rule = bmi_update_rule
        self.bmi.posterior_threshold = posterior_threshold
        self.bmi.pos_buffer_len = pos_buffer_len
        self.bmi.two_steps = two_steps_decoding
        self.laser_duration = laser_duration
        BMI.set_binner(self.bmi, bin_size=bsize, B_bins=Bbins)
        

        if self.bmi.binner is not None:
            try:
                self.bmi_func(mode, targetID, neuron_id, threshold, window_duration)
            except KeyboardInterrupt:
                print("Terminating the loop...")
            finally:
                self.bmi.close()

class GUIView:
    def load_spike(self, spike_file):
        df = pd.read_pickle(spike_file)
        n_unit = int(df['spike_id'].max())
        spike_time = np.zeros(n_unit, dtype=object)
        spike_fr = np.zeros(n_unit)
        spike_group = np.zeros(n_unit, dtype=int)

        duration = (df['frame_id'].iloc[-1] - df['frame_id'].iloc[0]) / 25000

        for i_unit in range(n_unit):
            in_unit = df['spike_id'] == (i_unit + 1)
            if sum(in_unit) == 0:
                continue
            spike_group[i_unit] = np.unique(df['group_id'][in_unit])[0]
            spike_time[i_unit] = df['frame_id'][in_unit].to_numpy() / 25000
            spike_fr[i_unit] = sum(in_unit) / duration

        return spike_group, spike_time, spike_fr

    def plot_spike_firing_rate_and_isi(self, spike_file, bin_size=1.0):
        _, spike_times, spike_fr = self.load_spike(spike_file)
        
        n_unit = len(spike_times)
        
        fig, axes = plt.subplots(n_unit, 3, figsize=(15, 5 * n_unit))
        
        for i_unit in range(n_unit):
            spike_time = spike_times[i_unit]
            
            # Plot Firing Rate
            
            axes[i_unit, 0].hist(spike_time, bins=50)
            axes[i_unit, 0].set_title(f"Unit {i_unit+1} - Firing Rate: {spike_fr[i_unit]:.2f} Hz")
            axes[i_unit, 0].set_xlabel("Time (s)")
            axes[i_unit, 0].set_ylabel("Spike Count")
            
            # Plot ISI distribution
            isi = np.diff(spike_time)
            axes[i_unit, 1].hist(isi, bins=50, color='red')
            axes[i_unit, 1].set_title(f"Unit {i_unit+1} - ISI Distribution")
            axes[i_unit, 1].set_xlabel("ISI (s)")
            axes[i_unit, 1].set_ylabel("Count")
            
            # Calculate and plot Firing Rate distribution in 1-second windows
            bin_count, bins = np.histogram(spike_time, np.arange(0, spike_time[-1] + bin_size, bin_size))
            
            # Plot FR distribution
            spike_count, spike_bins = np.histogram(bin_count, 50)
            smoothed_spike_count = gaussian_filter1d(spike_count, sigma=2)
            axes[i_unit, 2].plot(spike_bins[:-1], smoothed_spike_count,color='purple')
            axes[i_unit, 2].set_title(f"Unit {i_unit+1} - FR Distribution (1s windows)")
            axes[i_unit, 2].set_xlabel("Firing Rate (Hz)")
            axes[i_unit, 2].set_ylabel("Count")
            
            # Calculate and plot the 90th percentile line
            fr_90th_percentile = np.percentile(bin_count, 90)
            axes[i_unit, 2].axvline(fr_90th_percentile, color='green', linestyle='dashed', linewidth=1)
            axes[i_unit, 2].text(fr_90th_percentile, axes[i_unit, 2].get_ylim()[1] * 0.9,
                                f' 90th percentile: {fr_90th_percentile:.2f} Hz',
                                color='green', fontsize=12, ha='left')
        
        plt.tight_layout()
        plt.show()

    # example)
    # spike_file = 'path_to_spike_file'
    # gui_view = GUIView()
    # gui_view.plot_spike_firing_rate_and_isi(spike_file)
