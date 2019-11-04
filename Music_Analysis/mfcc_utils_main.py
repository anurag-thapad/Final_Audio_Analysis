# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:50:16 2019

Module for:
    1. all self implemented signal processing functions
    2. generate mfcc for single frame
@author: Satashree
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(x):
    """
    Plot the given signal
    """
    plt.figure(figsize=(20,5))
    plt.plot(x)
    plt.title("Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    
    

def hz_to_mel(hz):
    """
    Convert Hertz to Mel, using HTK formula
        Input:
            hz: Value in Hertz
        Output:
            Return value in Mel scale
    """
    return 2595*np.log10(1 + hz/700.0)


def mel_to_hz(mel):
    """
    Convert Mel to Hertz, using HTK formula
        Input:
            mel: Value in Mel scale
        Output:
            Return value in Hertz
    """
    return 700*(10**(mel/2595.0) - 1)


def preemphasis(sig, coeff=0.97):
    """
    Perform preemphasis on sig
        Input:
            sig: Array of amplitude values of signal
            coeff: Preemphasis coefficient
        Output:
            Return preemphasised signal
    """
    return np.append(sig[0], sig[1:]-coeff*sig[:-1])


def hamming_window(win_length):
    """
    Generate Hamming window
        Input:
            win_length: Length of the Hamming window
        Output:
            Return the Hamming window of desired length
    """
    n = np.arange(0,win_length)
    return 0.54-0.46*np.cos((2*np.pi*n)/(win_length-1))


def my_FFT(x):
     #https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
    """
    Return the Cooley-Tukey FFT of a signal
        Input:
            x: Signal
        Output: FFT of signal
        
    Note: Cooley-Tukey FFT requires that the length of FFT is a power of two. That has been taken
    care of in my_RFFT function, aand my_RFFT is dependent on my_FFT        
    """
    N = x.shape[0]
    if N<=1:
        return x
    
    X_even = my_FFT(x[0::2])
    X_odd = my_FFT(x[1::2])
    T = [np.exp(-2j*np.pi*k/N) * X_odd[k] for k in range(N//2)]
    
    ft = [X_even[k] + T[k] for k in range(N//2)] + [X_even[k] - T[k] for k in range(N//2)]
    return np.array(ft)


def my_RFFT(x, N=None):
    """
    Return N point RFFT of x
        Input:
            x: Signal
            N: Length of FFT, power of 2
        Ouput:
            Return the N point real FFT
    """
    # if N is smaller than length of input, input is cropped
    # if N is larger, input is padded with zeros 
    
    len_x = x.shape[0]
    
    if N<len_x:
        x = x[:N]
    elif N>len_x:

        z = np.zeros((N-len_x))
        x = np.append(x, z)
    
    return my_FFT(x)[:N//2+1]


def power_spectrum(frame, N=None):
    """
    Return the power spectrum/ magnitude spectrum of the frame
        Input:
            frame: Single frame or array of frames
            N = Length of FFT
        Output:
            Return the magnitude of frame
    """
    return (np.absolute(frame)**2)*(1.0/N)


def log_spectrum(spec):
    """
    Return the spectrum in log10 scale (i.e in Decible dB)
    """
    return 10*np.log10(spec)


def filter_bank(sr=None, nfilt=None, N=None):
    
    #http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    """
    Return the Mel filter bank of size(nfilt, N/2+1)
        Input:
            sr: Sampling rate of signal
            nfilt: No. of filters required
            N = Length of FFT
        Output:
            Mel filter bank
    """
    fbank = np.zeros((nfilt, N//2+1))
    
    low_freq = 0
    high_freq = sr/2
    low_freq_mel = hz_to_mel(low_freq)
    high_freq_mel = hz_to_mel(high_freq)
    
    # get nfilt+2 equally spaced points in Mel scal
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt+2)
    
    # convert all mel points back to Hertz
    hz_points = mel_to_hz(mel_points)
    
    # get corresponding FFT bin
    bin = np.floor((N+1)*hz_points/sr)
    
    for m in range(1, nfilt+1):
        f_m_minus_1 = int(bin[m-1])
        f_m = int(bin[m])
        f_m_plus_1 = int(bin[m+1])
        
        for k in range(f_m_minus_1, f_m):
            fbank[m-1, k] = (k - bin[m-1])/(bin[m] - bin[m-1])
        
        for k in range(f_m, f_m_plus_1):
            fbank[m-1, k] = (bin[m+1] - k)/(bin[m+1] - bin[m])
            
    return fbank


def my_DCT(x, num_coeff=20):
    """
    Return the type 2 orthonormalized DCT coefficients of x
        Input:
            x: Signal
            num_coeff: Number of coefficients required, be default calculates the first 20
        Output:
            Array of coefficients
    """
    N = x.size
    n = np.arange(0, N)
    c = np.zeros((num_coeff+1))
    q = x * np.cos(0)
    c[0] = np.sqrt(1/N) * np.sum(q)
    for k in range(1, num_coeff+1):
        p = x * np.cos(np.pi*k*(2*n + 1)/(2*N))
        c[k] = np.sqrt(2/N) * np.sum(p)
        
    return c
    

def my_deltas(feat, win):
    
    # http://www1.icsi.berkeley.edu/Speech/docs/HTKBook/node65_mn.html
    """
    Return the delta features
       Input:
           feat: Array of the features, of size (nframes, nfeatures)
           win:  Window length over which deltas are calculated
       Output:
           delta_feat: Deltas of feat, having same shape as feat
    """
    delta_feat = np.zeros(feat.shape)
    padded_feat = np.pad(feat.copy(), ((win, win),(0,0)), mode="edge")
    nframes = feat.shape[0]
    denominator = 2 * np.sum(np.square(np.arange(1, win+1)))
    
    for t in range(nframes):
        s = 0
        for n in range(1, win+1):
            s += n*(padded_feat[t+n+win] - padded_feat[t-n+win])
        delta_feat[t] = s/denominator
    
    return delta_feat


def single_frame_mfcc(frame, sr=None, N=None, nfilt=None, num_ceps=None):
    """
    Generate the required number of MFCCs for a single frame
        Input:
            frame: Given frame
            sr: Sampling rate of signal
            N: Length of FFT
            nfilt: No. of triangular overlapping Mel filters
            num_ceps: No. of MFC coefficients required
        Output:
            mfccs: Required no. of MFCCs
    """
    frame_len = len(frame)
    # Preemphasise the frame
    #frame = preemphasis(frame)
    # Hamming window
    frame *= hamming_window(frame_len)
    #print(frame)
    # Get RFFT
    spec = my_RFFT(frame, N=N)
    # Get power spectrum
    pow_frame = power_spectrum(spec.copy(), N=N)
    # Generate filter bank
    fbank = filter_bank(sr=sr, nfilt=nfilt, N=N)
    # Get Mel spectrograme
    mel_spec = np.dot(pow_frame.copy(), fbank.T)
    mel_spec = np.where(mel_spec==0, np.finfo(float).eps, mel_spec)
    # Log power Mel spectrogram
    mel_spec_db = log_spectrum(mel_spec.copy())
    # Take DCT of Log mel spectrogram
    mfccs = my_DCT(mel_spec_db.copy())
    # Keep required number of coeff
    mfccs = mfccs[:num_ceps]
    
    return mfccs
    
    
####################################################################################################### 
    