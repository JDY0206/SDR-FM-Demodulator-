from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox 
import pyqtgraph as pg 
import numpy as np
import time 
from rtlsdr import RtlSdr
import signal, sys
import sounddevice as sd
import scipy.signal as spsig
import queue 
from math import gcd

#----------------------
# Default Settings
#----------------------
fft_size = 4096 # Buffer size
num_rows = 200
center_freq = 101e6
sample_rates = [0.24, 0.25, 1, 1.024, 1.5, 2, 2.048, 2.5, 2.8, 3.2] # MHz
sample_rate = 1.024e6
time_plot_samples = 500
gain = 50 # 0 to 73 dB
chunk_samples = 262144
iq_queue = queue.Queue(maxsize=256)

#----------------------
#Initalize SDR
#----------------------
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq 
sdr.gain = 'auto' 
sdr.set_direct_sampling(0) # Disable direct sampling to avoid HF

#----------------------
# Background Thread Set-up
#----------------------

class AudioWorker(QObject):
    def __init__ (self, sdr, sample_rate, center_freq, audio_rate = 48000):
        super().__init__()
        self.iq_queue = iq_queue
        self.sample_rate = float(sample_rate)
        self.center_freq = center_freq
        self.audio_rate = float(audio_rate)
        self.running = True 
        self.rms_smooth = 1e-6
        self.prev_sample = 0+0j
    
    def lowpass_filter(self, data, cutoff=15000, fs=48000, order=5):
        nyq = 0.5 * fs 
        normal_cutoff = cutoff / nyq 
        b, a = spsig.butter(order, normal_cutoff, btype = 'low', analog=False) 
        filtered_data = spsig.lfilter(b, a, data)
        return filtered_data
        
    def bandpass_filter(self, data, lowcut=87e3, highcut=108e3, fs=2.5e5, order=5):
        nyq = 0.5 * fs 
        low = lowcut / nyq 
        high = highcut / nyq 
        b, a = spsig.butter(order, [low,high], btype='band')
        return spsig.lfilter(b, a, data)
    
    def highpass_filter(self, data, cutoff=50, fs=48000, order=1):
        nyq = 0.5* fs 
        normal_cutoff = cutoff / nyq 
        b, a = spsig.butter(order, normal_cutoff, btype='high', analog=False)
        return spsig.lfilter(b, a, data)
    
    def noise_blanker(self, audio, threshold = 0.5):
        diff = np.abs(np.diff(audio, prepend=audio[0]))
        audio[diff > threshold] = 0.0
        return audio 
        
    def run(self):
        expected_audio = int(round(fft_size * (self.audio_rate / self.sample_rate)))
        # Open Audio Stream
        stream = sd.OutputStream(
            samplerate = self.audio_rate,
            channels = 1,
            dtype = 'float32',
            blocksize=1024,
            latency='low' )  
        stream.start()
        
        g = gcd(int(self.audio_rate), int(self.sample_rate))
        up = int(self.audio_rate // g)
        down = int(self.sample_rate // g)
        
        while self.running:
            try:
                samples = self.iq_queue.get(timeout=0.5) # Wait for IQ
            except queue.Empty:
                continue 
            if samples is None or samples.size < 2:
                continue
        
            # if self.iq_queue.qsize() < 4:
                # print("Warning: IQ Queue running low")  
                        
            # Remove DC from IQ first
            samples = samples - np.mean(samples)

            # FM Demodulation using unwrapped phase difference
            combined = np.concatenate(([self.prev_sample], samples))
            phase = np.unwrap(np.angle(combined))
            angle = np.diff((phase))
            self.prev_sample = samples[-1]

            # Scale deviation to prevent clipping
            angle = angle / np.std(angle) * 0.5

            # Lowpass before resample
            angle = self.lowpass_filter(angle, cutoff=15000, fs=self.sample_rate, order=5)

            # Resample (direct ratio instead of gcd trick)
            audio = spsig.resample_poly(angle, int(self.audio_rate), int(self.sample_rate), window=('kaiser', 8.6))
            
            # Highpass Filter
            audio = self.highpass_filter(audio, cutoff=40, fs=self.audio_rate, order=2)
            
            # Extra smoothing Low Pass at audio rate
            audio = self.lowpass_filter(audio, cutoff=10000, fs=self.audio_rate, order=6)

            # De-emphasis filter 
            tau = 75e-6
            b = [1]
            a = [1, -np.exp(-1/(self.audio_rate*tau))]
            alpha = np.exp(-1.0/(self.audio_rate*tau))
            audio = spsig.lfilter(b, a, audio)
            
            # Squelch
            gate_threshold = 0.005
            if np.std(audio) < gate_threshold:
                audio[:] = 0.0
            
            # Noise Blanker
            audio = self.noise_blanker(audio)
            
            # Smoothed RMS normalization
            cur_rms = np.sqrt(np.mean(audio*audio) + 1e-12)
            self.rms_smooth = 0.995*self.rms_smooth + 0.005*cur_rms
            if self.rms_smooth > 1e-7:
                audio = audio / (4*self.rms_smooth)
            
            # Remove extreme values before writing 
            audio = np.clip(audio, -1.0, 1.0)
            
            if len(audio) % stream.blocksize != 0:
                pad_len = stream.blocksize - (len(audio) % stream.blocksize)
                audio = np.pad(audio, (0, pad_len), mode='constant')
            
            # Send to sound device 
            try:
                stream.write(audio.astype(np.float32))
            except Exception as e:
                print("Audio write error:", e)
                continue
            
        stream.stop()
        stream.close()
        
    def stop(self):
        self.running = False
        try:
            self.iq_queue.put_nowait(None)
        except queue.Full:
            pass
            
class SDRWorker(QObject):
    def __init__(self):
        super().__init__()
        self.gain = gain
        self.sample_rate = sample_rate 
        self.freq = center_freq
        self.spectrogram = -50*np.ones((fft_size, num_rows))
        self.PSD_avg = -50*np.ones(fft_size)
        self.pending_gain = None 
        self.pending_sample_rate = None
        self.pending_freq = None
        self.running = True
        
    # PyQt Signals 
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal()
    
    #PyQt Slots
    def update_freq(self, val): # Could also just modify SDR in GUI Thread
        print("Scheduled frequency update to:", val, 'kHz')
        self.pending_freq = val
        
    def update_gain(self, val):
        print("Scheduled gain update to:", val, 'dB')
        self.pending_gain = val
    
    def update_sample_rate(self,val):
        print("Scheduled sample rate update to:", sample_rates[val], 'MHz')
        self.pending_sample_rate = sample_rates[val] * 1e6
            
    #----------------------
    # Main Loop
    #----------------------
    def run(self):
        if not self.running:
            return
        
        start_t = time.time()
        
        # Apply pending SDR updates to avoid crash while reading
        if self.pending_gain is not None:
            try:
                sdr.gain = 'auto'
            except OSError as e:
                print("Failed to set gain:", e)
            self.pending_gain = None
        if self.pending_freq is not None:
            try:
                sdr.center_freq = self.pending_freq 
            except OSError as e:
                print("Failed to set frequency:", e)
            self.pending_freq = None 
        if self.pending_sample_rate is not None:
            try:
                sdr.sample_rate = self.pending_sample_rate 
            except OSError as e:
                print("Failed to set sample rate:", e)
            self.pending_sample_rate = None
              
        # Grab some samples from RTL-SDR
        samples = np.array(sdr.read_samples(chunk_samples), dtype=np.complex64)
        
        # Replace NaN/ Inf with zeros
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            iq_queue.put(samples, timeout=1)
        except queue.Full:
                iq_queue.get_nowait() 
                iq_queue.put(samples, timeout=1)
        
        # Update Time Plot
        self.time_plot_update.emit(samples[0:time_plot_samples])
        
        fft_block = samples[:fft_size]
        # Compute PSD
        PSD = np.abs(np.fft.fftshift(np.fft.fft(fft_block))**2/fft_size)
        PSD = 10.0 * np.log10(np.maximum(PSD, 1e-12)) # Avoid log10(0)
    
        # Update averaged PSD for freq plot
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)
        
        # Update Waterfall
        if self.spectrogram.shape[0] != PSD.shape[0]:
            self.spectrogram = np.zeros((PSD.shape[0], self.spectrogram.shape[1]))
            
        self.spectrogram = np.roll(self.spectrogram, -1, axis=1) # Shifts waterfall 1 row
        self.spectrogram[:,-1] = PSD # Fill last row with new FFT results
        self.waterfall_plot_update.emit(self.spectrogram)
        
        # elapsed = time.time() - start_t
        # if elapsed > 0:
          #   print("Frames per second:", 1/(elapsed))
        #print("SDR:", sdr.center_freq, sdr.sample_rate, "Worker:", window.worker.freq, window.worker.sample_rate)

        self.end_of_run.emit() # Emit the signal to keep the loop going
        
#----------------------
# Application Layout/Inner Workings
#----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.audio_thread = QThread()
        self.audio_worker = AudioWorker(sdr, sample_rate, center_freq)
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_thread.started.connect(self.audio_worker.run)
        self.audio_thread.start()
        
        self.sdr_thread = QThread()
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)
        
        self.setWindowTitle("Spectrum Analyzer")
        self.setFixedSize(QSize(1000, 600)) # Window Size 
        
        self.spectrogram_min = 0
        self.spectrogram_max = 0
        
        layout = QGridLayout() # Overall Layout
        
        # Time Plot 
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time[microseconds]'})
        time_plot.setMouseEnabled(x=False, y=True)
        time_plot.setYRange(-1.1, 1.1)
        time_plot_curve_i = time_plot.plot([])
        time_plot_curve_q = time_plot.plot([])
        layout.addWidget(time_plot, 1, 0)
        
        # Time Plot Auto Range Buttons
        time_plot_auto_range_layout = QVBoxLayout()
        layout.addLayout(time_plot_auto_range_layout, 1, 1)
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : time_plot.autoRange())
        time_plot_auto_range_layout.addWidget(auto_range_button)
        auto_range2 = QPushButton('-1 to +1\n(ADC Limits)')
        auto_range2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
        time_plot_auto_range_layout.addWidget(auto_range2)
        
        # Freqency Plot 
        freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
        freq_plot.setMouseEnabled(x=False, y=True)
        freq_plot_curve = freq_plot.plot([])
        freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        freq_plot.setYRange(-30,20)
        layout.addWidget(freq_plot, 2, 0)
        
        # Freqency Auto Range Button
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : freq_plot.autoRange())
        layout.addWidget(auto_range_button)
        
        # Layout Container for Waterfall 
        waterfall_layout = QHBoxLayout()
        layout.addLayout(waterfall_layout, 3, 0)
        
        # Waterfall Plot 
        waterfall = pg.PlotWidget(labels={'left': 'Time[s]', 'bottom': 'Frequency [MHz]'})
        imageitem = pg.ImageItem(axisOrder = 'col-major')
        waterfall.addItem(imageitem)
        waterfall.setMouseEnabled(x=False, y=False)
        waterfall_layout.addWidget(waterfall)
        
        # Colorbar for Waterfall
        colorbar = pg.HistogramLUTWidget()
        colorbar.setImageItem(imageitem) # Connects bar to waterfall imageitem
        colorbar.item.gradient.loadPreset('viridis') # Set the color map and image item
        imageitem.setLevels((-30, 20))
        waterfall_layout.addWidget(colorbar)
        
        # Waterfall Auto Range Button
        auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
        def update_colormap():
            imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
        auto_range_button.clicked.connect(update_colormap)
        layout.addWidget(auto_range_button, 3, 1)
        
        # Frequency Slider with label
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setMinimum(24) # 24 MHz 
        freq_slider.setMaximum(200) # 200 MHz
        freq_slider.setValue(100)
        freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        def slider_to_worker(val):
            freq_hz = val * 1_000_000 # Convert MHz to Hz
            self.worker.update_freq(freq_hz) 
        freq_slider.sliderMoved.connect(slider_to_worker)
        freq_label = QLabel()
        def update_freq_label(val):
            freq_label.setText(f"Frequency [MHz]: {val:.3f}")
            freq_plot.autoRange()
        freq_slider.sliderMoved.connect(update_freq_label)
        update_freq_label(freq_slider.value()) # Initalize the label
        layout.addWidget(freq_slider, 4, 0)
        layout.addWidget(freq_label, 4, 1)
        
        # Fine Tune Frequency Slider with Label (kHz)
        fine_freq_slider = QSlider(Qt.Orientation.Horizontal)
        fine_freq_slider.setMinimum(-1000) 
        fine_freq_slider.setMaximum(1000) 
        fine_freq_slider.setValue(100)
        fine_freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        fine_freq_label = QLabel()
        def update_fine_label(val):
            fine_freq_label.setText(f"Fine Tune Frequency [kHz]: {val:.3f}")
            new_freq = window.worker.freq + val * 1_000
            window.worker.update_freq(new_freq)
        fine_freq_slider.sliderMoved.connect(update_fine_label)
        update_freq_label(fine_freq_slider.value()) # Initalize the label
        layout.addWidget(fine_freq_slider, 5, 0)
        layout.addWidget(fine_freq_label, 5, 1)
        
        # Gain Slider with Label
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 73)
        gain_slider.setValue(gain)
        gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        gain_slider.setTickInterval(2)
        gain_slider.sliderMoved.connect(self.worker.update_gain)
        gain_label = QLabel()
        def update_gain_label(val):
            gain_label.setText("Gain: " + str(val))
        gain_slider.sliderMoved.connect(update_gain_label)
        update_gain_label(gain_slider.value()) # Initialize the label
        layout.addWidget(gain_slider, 6, 0)
        layout.addWidget(gain_label, 6, 1)
        
        # Sample Rate Dropdown Box
        sample_rate_box = QComboBox()
        sample_rate_box.addItems([str(x) + 'MHz' for x in sample_rates])
        sample_rate_box.setCurrentIndex(0)
        sample_rate_box.currentIndexChanged.connect(self.worker.update_sample_rate)
        sample_rate_label = QLabel()
        def update_sample_rate_label(val):
            sample_rate_label.setText(f"Sample Rate: {sample_rates[val]} MHz")
        sample_rate_box.currentIndexChanged.connect(update_sample_rate_label)
        update_sample_rate_label(sample_rate_box.currentIndex())
        layout.addWidget(sample_rate_box, 7, 0)
        layout.addWidget(sample_rate_label, 7, 1)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Signals and Slots 
        def time_plot_callback(samples):
            time_plot_curve_i.setData(samples.real)
            time_plot_curve_q.setData(samples.imag)
        
        def freq_plot_callback(PSD):
            # Ensure PSD is 1D
            PSD_1D = PSD.ravel()
            freq_axis = np.linspace(
                window.worker.freq - window.worker.sample_rate/2, 
                window.worker.freq + window.worker.sample_rate/2, 
                len(PSD_1D)
            ) / 1e6 # Convert Hz to MHz
            
            freq_plot_curve.setData(freq_axis, PSD_1D)
            freq_plot.setXRange(freq_axis[0], freq_axis[-1])
        
        def waterfall_plot_callback(spectrogram):
            spectrogram_2d = np.atleast_2d(spectrogram)
            imageitem.setImage(spectrogram_2d, autoLevels=False)
            sigma = np.std(spectrogram_2d)
            mean = np.mean(spectrogram_2d)
            self.spectrogram_min = mean - 2*sigma
            self.spectrogram_max = mean + 2*sigma
            
        def end_of_run_callback():
            QTimer.singleShot(0, self.worker.run) # Run worker again immediately
            
        # Connect signals to callbacks    
        self.worker.time_plot_update.connect(time_plot_callback)
        self.worker.freq_plot_update.connect(freq_plot_callback)
        self.worker.waterfall_plot_update.connect(waterfall_plot_callback)
        self.worker.end_of_run.connect(end_of_run_callback)
        
        self.sdr_thread.started.connect(self.worker.run) # Starts worker thread
        self.sdr_thread.start()
        
    def closeEvent(self, event):
        self.sdr_thread.quit()
        self.sdr_thread.wait()
        self.audio_worker.stop()
        self.audio_thread.quit()
        self.audio_thread.wait()
        try:
            iq_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            sdr.close()
        except Exception as e:
            print("Error closing SDR:", e)
        event.accept()

def handle_close(sig, frame):
    print("Closing SDR and exiting")
    try:
        sdr.close()
    except Exception as e:
        print("Error closing SDR:", e)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_close)

app = QApplication([])
window = MainWindow()
window.show()
app.exec() # Start Event Loop
        
    

        
