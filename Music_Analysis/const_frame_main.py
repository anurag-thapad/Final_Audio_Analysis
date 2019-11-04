# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:46:27 2019

@author: Satashree
"""

import mfcc_utils_main
from mfcc_utils_main import np

def frame_signal(sig, sr=None, frame_size=None, frame_stride=None):
    """
    Divide the signal into overlapping frames of constant frame size and witha constant stride, such that
    all the frames have equal number of samples.
         Input:
             sig: List of the amplitude values of the signal
             sr: Sampling rate of the signal
             frame_size = Size of each frame, in terms of seconds
             frame_stride = Seconds after which next frame starts
    """
    frame_length = int(frame_size*sr)
    frame_step = int(frame_stride*sr)
    sig_length = len(sig)
    if sig_length < frame_length:
        nframes = 1
    else:
        nframes = 1 + int(np.ceil((1.0*sig_length - frame_length)/frame_step))
        pad_sig_length = (nframes-1)*frame_step + frame_length
        z = np.zeros((pad_sig_length - sig_length))
        padded_sig = np.append(sig.copy(), z)
        
    # Get indices of the frames
    a = np.arange(0,frame_length)
    b = np.arange(0, nframes*frame_step, frame_step)
    indices = np.add.outer(a,b).T
    frames = padded_sig[indices.astype(np.int32, copy=False)]
    
    return frames, nframes



def sig_mfcc(sig, sr=None, frame_size=None, frame_stride=None, N=512, nfilt=40, num_ceps=13):
    # Using self implemented DSP functions
    """
    Generate the MFCC of the given signal. Frames the signal as well.
        Input:
            sig: List of the amplitude values of the signal
            sr: Sampling rate of the signal
            frame_size = Size of each frame, in terms of seconds
            N = Length of the FFT
            nfilt = No. of overlapping triangular Mel filter bank
            num_ceps = No. of MFC coeffcients required
        Ouput:
            mfccs: Required number of MFCCs
    """
    frame_len = int(frame_size*sr)
    # Preemphasise the signal
    pre_emp_sig = mfcc_utils_main.preemphasis(sig.copy())
    # Frame the signal
    frames, nframes = frame_signal(pre_emp_sig.copy(), sr=sr, frame_size=frame_size, frame_stride=frame_stride)
    # Windowing
    frames *= mfcc_utils_main.hamming_window(frame_len)
    # Get RFFT
    spec_frame = np.apply_along_axis(mfcc_utils_main.my_RFFT, 1, frames.copy(), N=512) #axis 1
    # Get power spectrum
    pow_frame = mfcc_utils_main.power_spectrum(spec_frame.copy(), N=N)
    # Generate filter bank
    fbank = mfcc_utils_main.filter_bank(sr=sr, nfilt=nfilt, N=N)
    # Get Mel spectrogram
    mel_spec = np.dot(pow_frame.copy(), fbank.T)
    mel_spec = np.where(mel_spec==0, np.finfo(float).eps, mel_spec)
    # Log power Mel spectrogram
    mel_spec_db = 10*np.log10(mel_spec.copy())
    # Take DCT of Log mel spectrogram
    mfccs = np.apply_along_axis(mfcc_utils_main.my_DCT, 1, mel_spec_db)
    # Keep required number of coeff
    mfccs = mfccs[:, :num_ceps+1]
    #print(mfccs.shape)
    return mfccs, nframes

##################################################################################################################

