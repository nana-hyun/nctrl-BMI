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
from tuning import Tuning
from scipy.ndimage import gaussian_filter1d

class BMIRealtime:
    def __init__(self, prb, fetfile, ttlport):
        self.bmi = BMI(prb=prb, fetfile=fetfile, ttlport=ttlport)
    
    def plot_raster(self, target_neuron_id, bmi_output, window_duration, spike_times):
        t = bmi_output.timestamp  # time
        spk_id = bmi_output.spk_id
        rm_t = t % 1000

        if spk_id == target_neuron_id:
            if len(spike_times) != 0:
                if spike_times[-1] > rm_t:
                    plt.close()
                    spike_times.clear()
            spike_times.append(rm_t)

        clear_output(wait=True)
        plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8)
        plt.xlim(0, 1000)
        plt.xlabel('Time (s)')
        plt.title('Real-time Raster Plot for Neuron {}'.format(target_neuron_id))
        plt.show()
        time.sleep(0.1)
        return spike_times

    def send_signal_to_teensy(self):
        if self.bmi.TTLserial is not None:
            self.bmi.TTLserial.write(b'a')
            self.bmi.TTLserial.flush()

    def reset_signal_to_teensy(self):
        if self.bmi.TTLserial is not None:
            self.bmi.TTLserial.write(b'b')
            self.bmi.TTLserial.flush()
    
    def bmi_func(self, mode, targetID, neuron_id, threshold, window_duration):
        spike_times = []
        while True:
            frFlag = 0
            spkFlag = 0
            bmi_output = self.bmi.read_bmi()
            spike_times = self.plot_raster(targetID, bmi_output, window_duration, spike_times)
            
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
                           bmi_update_rule, posterior_threshold, two_steps_decoding, mode, targetID, window_duration):
        pos_buffer_len = int(float(t_smooth) / float(bsize))
        self.bmi.bmi_update_rule = bmi_update_rule
        self.bmi.posterior_threshold = posterior_threshold
        self.bmi.pos_buffer_len = pos_buffer_len
        self.bmi.two_steps = two_steps_decoding
        BMI.set_binner(self.bmi, bin_size=bsize, B_bins=Bbins)

        if self.bmi.binner is not None:
            try:
                self.bmi_func(mode, targetID, neuron_id, threshold, window_duration)
            except KeyboardInterrupt:
                print("Terminating the loop...")
            finally:
                self.bmi.close()

class GUIView:
    def plot_spike_firing_rate_and_isi(self, spike_file):
        t = Tuning()
        t.load_spike(spike_file)
        
        n_unit = len(t.spike_time)
        
        fig, axes = plt.subplots(n_unit, 2, figsize=(10, 5 * n_unit))
        
        for i_unit in range(n_unit):
            spike_time = t.spike_time[i_unit]
            
            # Plot Firing Rate
            duration = spike_time[-1] - spike_time[0]
            firing_rate = len(spike_time) / duration
            
            axes[i_unit, 0].hist(spike_time, bins=50)
            axes[i_unit, 0].set_title(f"Unit {i_unit+1} - Firing Rate: {firing_rate:.2f} Hz")
            axes[i_unit, 0].set_xlabel("Time (s)")
            axes[i_unit, 0].set_ylabel("Spike Count")
            
            # Plot ISI distribution
            isi = np.diff(spike_time)
            axes[i_unit, 1].hist(isi, bins=50, color='red')
            axes[i_unit, 1].set_title(f"Unit {i_unit+1} - ISI Distribution")
            axes[i_unit, 1].set_xlabel("ISI (s)")
            axes[i_unit, 1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()
