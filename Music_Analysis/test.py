# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:20:24 2019

@author: Satashree
"""

import librosa
import pitch_sync_frame_main

fp = 'C:/Users/Satashree/Desktop/music_analysis/daisy1.wav'
sig, sr = librosa.core.load(fp)

s, pitch = pitch_sync_frame_main.signal_start_point(sig, offset=4)
marks, pitches, num_frames = pitch_sync_frame_main.pitch_marks(sig, pitch, offset=4)

print(pitches)