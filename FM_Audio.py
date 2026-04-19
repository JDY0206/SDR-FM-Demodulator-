import numpy as np
import time 
from rtlsdr import RtlSdr 
import signal, sys 
import sounddevice as sd 
import scipy.signal as spsig 
from scipy.signal import resample_poly, firwin, lfilter 
import queue 
import threading 

#--------------------
# Set Defaults 
#--------------------
sampleRate = 2.048e6   # 2.048 MHz
centerFreq = 100.7e6   # Center frequency should be in Hz
Audioblock = 1024      # Frame for feeding audio playback

# Add in pushing into queue for external sources
plot_queue = None # Can be replaced by an external queue
iq_queue = None 

# De-emphasis Filter coefficients 
tau = 75e-6
x = np.exp(-1 / (48000 * tau))
b = [1 - x]
a = [1, -x]

#--------------------
# Initalize SDR 
#--------------------
sdr = RtlSdr() 
sdr.sample_rate = sampleRate
sdr.center_freq = centerFreq 
sdr.gain = 'auto' 

#--------------------
# Audio Output Queue and Stream
#--------------------
q = queue.Queue(maxsize = 64) 

def caller(outdata, frames, time, status): 
    if not q.empty():
        data = q.get()
        if len(data) < frames: # Pad if too short 
            data = np.pad(data, ((0, frames - len(data)), (0, 0)))
        elif len(data) > frames: # Trim if too long
            data = data[:frames]
        outdata[:] = data 
    else: 
        outdata[:] = np.zeros((frames, 1)) 

stream = sd.OutputStream(
    samplerate = 48000, 
    channels = 1, 
    blocksize = 1024, 
    callback = caller
    ) 

stream.start() 

#--------------------
# Predefined Filters 
#--------------------
# Lowpass for audio with a 15kHz cutoff at SDR rate 
lp_taps = firwin(101, 15e3, fs=sampleRate/8) 

# Channel limit around station of about 100 kHz 
channelTaps = firwin(129, 100e3, fs = sampleRate)

stop_event = threading.Event()

def stopper(sig, frame):
    print("Stop signal received")
    stop_event.set() 
    sdr.cancel_read_async()

signal.signal(signal.SIGINT, stopper)

#--------------------
# FM Radio Processing Block 
#--------------------

# Demod and feed audio queue
def FMRadio(samples, sdr):
    
    if stop_event.is_set():
        raise KeyboardInterrupt 
        return  
    
    # Remove DC from IQ 
    samples = samples - np.mean(samples)
    
    # Send raw IQ samples into queue for visuals
    if iq_queue is not None:
        try:
            iq_queue.put_nowait(samples[:4096].copy())
        except queue.Full:
            pass
    
    # Channel Filter: bandlimit around station
    samples = lfilter(channelTaps, [1.0], samples)
   
    # FM Demod (Phase difference demod)
    audio = 0.5 * np.angle(samples[0:-1] * np.conj(samples[1:]))
    
    # Downsample by 8, 2.048 MHz to 256kHz 
    # Decimate includes a lowpass filter for anti-aliasing before downsampling
    audio = spsig.decimate(audio, 8, ftype = 'fir')
    decimatedRate = sampleRate/8
    
    # Lowpass Filter at 15 kHz designed for 256 kHz after downsampling
    audio = lfilter(lp_taps, [1.0], audio) 
    
    # Resample to 48 kHz (3/16)
    audio = resample_poly(audio, 48000, int(decimatedRate)) 
    
    # De-emphasis Filter since station pre-emphasizes
    # 75µs for North America
    audio = lfilter(b, a, audio) 
    
    # Normalize to a good listening level
    audio /= (np.std(audio) + 1e-6) * 3
    
#--------------------
# Visualization Block
#-------------------- 
    # Send small piece of data to visualizer 
    if plot_queue is not None: 
        try: 
            plot_queue.put_nowait(audio[:4096])
        except queue.Full:
            pass
    
    # Break into fixed chunks
    for i in range(0, len(audio), Audioblock):
        chunk = audio[i:i+Audioblock] 
        if len(chunk) < Audioblock:
            chunk = np.pad(chunk, (0, Audioblock - len(chunk)))
        try: 
            q.put(chunk.reshape(-1, 1)) 
        except queue.Full: 
            pass 


# Made a change from read_samples to read_samples_async
# Read_samples_async() lets the SDR driver push samples continuously into the processing callback 
# Avoids large blocking reads 
# Pipeline:
# Read IQ -> Demod -> Filter -> Resample -> queue 

# RTL-SDR operates at it's own thread 

# Changed center freq to be in Hz with e6 
# Resampled with 3/16 instead of large integers, just reduced to gcd 
# Decimated by 8 instead of 10 using the decimate function 
# Lowpass filter with the right sample rate (256000 not 2.048e6)
# Have to make sure the filter freq matches the sample rate 
# Deemphasis applied at audio rate after resampling to 48 kHz, not before at SDR rate 
# Precomputed the lowpass filter to reduce CPU strain

#--------------------
# Pipeline is as follows:
#--------------------
# Remove DC from samples
# Channel filter (Bandlimit around desired station)
# FM Demodulation 
# Decimate 
# Lowpass Filter
# Resample
# De-emphasis
# Normalize 
# Chop up the audio into blocks to feed the queue so that the sd is happy

