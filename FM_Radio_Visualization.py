'''
========================
SDR FM Radio Visualizer
========================
Features:
1. Waterfall (audio spectrogram, time x frequency)
2. Time-domain waveform
3. Frequency-domain spectrum with SNR overlay
4. RF Spectrum showing raw IQ from SDR
5. Manual Frequency textbox (MHz)
6. Dropdown of local Austin-area FM stations
7. Station name display 
8. Pause / Resume button
9. Signal strength meter
'''

import sys
import queue
import numpy as np
import FM_Audio
from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, 
    QProgressBar, QComboBox, QFrame
)
from PyQt6.QtGui import QCursor
import pyqtgraph as pg

#--------------------
# Set Defaults 
#--------------------
audio_rate = 48000
sdr_rate = 2.048e6
rf_fft_size = 4096
audio_fft_size = 2048
wf_rows = 180
wf_cols = 512

# Austin area FM stations (Mhz: "Call-sing - Name")
Local_stations = {
    88.7: "KAZI - Community Radio",
    89.5: "KGSR - Roots / Americana",
    90.5: "KUT  - NPR News & Music",
    91.7: "KMFA - Classical",
    93.3: "KLBJ - Classic Rock",
    93.7: "KVUE - News Talk",
    95.5: "KROX - Alternative",
    96.7: "KAMX - Mix",
    98.1: "KVET - Country",
    98.9: "KASE - Country",
    99.7: "KFMK - Classical / Smooth Jazz",
    100.7: "KRMH - Rhythmic",
    101.5: "KBPA - Jazz / Blues",
    103.5: "KLZT - Tejano",
    104.3: "KDHT - Dance / Pop",
    105.3: "KLBJ - Talk",
    106.3: "KIOL - Hot AC",
    107.1: "KEYI - Spanish Pop",
}

pg.setConfigOptions(antialias=True, background="#0a0f1a", foreground="#c8d8f0")

#--------------------
# Waterfall Ring Buffer
#--------------------
class WaterfallBuffer: 
    def __init__(self, n_rows = wf_rows, n_cols = wf_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols 
        self.buf = np.full((n_rows, n_cols), -80.0, dtype = np.float32)
    
    def push(self, spectrum_db):
        row = np.interp(
            np.linspace(0, len(spectrum_db) - 1, self.n_cols),
            np.arange(len(spectrum_db)),
            spectrum_db
        ).astype(np.float32)
        self.buf = np.roll(self.buf, 1, axis = 0)
        self.buf[0] = row 
        
    def get(self):
        return self.buf 
    
#--------------------
# DSP Helper Functions
#--------------------
def rf_fft(iq_samples, n = rf_fft_size):
    chunk = iq_samples[:n]
    windowed = chunk * np.blackman(n)
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n = n))
    power = np.abs(spectrum) **2
    power_db = 20 * np.log10(power / np.max(power) + 1e-12)
    return np.clip(power_db, -80, 0)

# One sided power spectrum of audio in dB
def audio_fft(audio, n = audio_fft_size):
    windowed = audio * np.hanning(len(audio))
    fft = np.fft.rfft(windowed, n = n)
    power_db = 20 * np.log10(np.abs(fft) + 1e-12) 
    return np.clip(power_db, -80, 0)

# Broadband SNR 300 to 2400 Hz (voice/music energy)
# Noise floor = 15,000 to 20,000Hz
def estimate_snr(audio, n = audio_fft_size):
    fft = np.fft.rfft(audio, n = n)
    freqs = np.fft.rfftfreq(n, d=1.0 / audio_rate)
    power = np.abs(fft) ** 2
    
    sig_m = (freqs >=300) & (freqs <= 3400) 
    nse_m = (freqs >= 15000) & (freqs <= 20000) 
    
    Ps = np.mean(power[sig_m]) if sig_m.any() else 1e-12
    Pn = np.mean(power[nse_m]) if nse_m.any() else 1e-12
    return float(np.clip(10 * np.log10(Ps / (Pn + 1e-12)), 0, 45))

# Return the frequency (Hz) of the dominant spectral peak in the audio
def peak_freq(audio, n = audio_fft_size):
    fft = np.fft.rfft(audio, n = n)
    freqs = np.fft.rfftfreq(n, d = 1.0 / audio_rate)
    mask = freqs > 200 # Ignore DC and sub-base
    idx = np.argmax(np.abs(fft[mask]))
    return float(freqs[mask][idx])

#--------------------
# SDR Worker Thread
#--------------------
class SDRThread(QThread):
    error = pyqtSignal(str)
    
    def __init__(self, sdr, parent = None):
        super().__init__(parent)
        self.sdr = sdr 
        
    def run(self):
        try:
            self.sdr.read_samples_async(FM_Audio.FMRadio, 256 * 1024)
        except Exception as exc:
            self.error.emit(str(exc))
    
    def stop(self):
        FM_Audio.stop_event.set()
        try:
            self.sdr.cancel_read_async() 
        except Exception:
            pass
        
#--------------------
# Main Visual Window
#--------------------
class SDRVisualizer(QMainWindow):
    
    poll_ms = 40  # About 25 fps
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR FM Radio")
        self.resize(1400, 900)
        
        # Shared queues 
        self.audio_queue = queue.Queue(maxsize = 32) 
        self.iq_queue = queue.Queue(maxsize = 32)
        FM_Audio.plot_queue = self.audio_queue
        FM_Audio.iq_queue = self.iq_queue
        
        # Define variable and functions 
        self.latest_audio = np.zeros(audio_fft_size)
        self.wf_buf = WaterfallBuffer()
        self.snr_history = np.zeros(200)
        self.paused = False 
        self.current_Mhz = FM_Audio.centerFreq / 1e6
        
        self.build_ui()
        self.start_sdr()
        
        self.timer = QTimer()
        self.timer.setInterval(self.poll_ms)
        self.timer.timeout.connect(self.poll) 
        self.timer.start() 
        
    #--------------------
    # Build UI 
    #--------------------
    def build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget) 
        root = QVBoxLayout(root_widget) 
        root.setContentsMargins(10, 10, 10, 6) 
        root.setSpacing(6) 
        
        root.addLayout(self.build_controls())
        root.addWidget(self.build_station_bar())
        
        # Top row: RF Specturum (left), Audio Freq Spectrum (right)
        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.addWidget(self.make_rf_spectrum())
        top_split.addWidget(self.make_audio_spectrum())
        top_split.setSizes([720, 720])
        
        # Bottom row: Waterfall (left), Waveform (right)
        bot_split = QSplitter(Qt.Orientation.Horizontal)
        bot_split.addWidget(self.make_waterfall())
        bot_split.addWidget(self.make_waveform())
        bot_split.setSizes([860, 580])
        
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(top_split)
        vsplit.addWidget(bot_split)
        vsplit.setSizes([380, 420])
        root.addWidget(vsplit)
        
        # Status Bar
        status_row = QHBoxLayout()
        self.status_label = QLabel("Connecting")
        
        self.rssi_bar = QProgressBar() 
        self.rssi_bar.setRange(0, 45)
        self.rssi_bar.setValue(0)
        self.rssi_bar.setTextVisible(True)
        self.rssi_bar.setFormat("SNR %v dB")
        self.rssi_bar.setFixedWidth(220)
        self.rssi_bar.setStyleSheet(            
            "QProgressBar { border:1px solid #1a2535; border-radius:3px; "
            "               background:#0a0f1a; color:#c8d8f0; }"
            "QProgressBar::chunk { background: qlineargradient("
            "  x1:0,y1:0,x2:1,y2:0, stop:0 #1a6b3a, stop:0.6 #2ecc71, stop:1 #f1c40f); }"
        )
        
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(QLabel("Signal:"))
        status_row.addWidget(self.rssi_bar)
        root.addLayout(status_row)
        
    #--------------------
    # Build Interactive Controls
    #--------------------
    def build_controls(self):
        bar = QHBoxLayout() 
        bar.setSpacing(8)
        
        # Frequency Changing Box
        self.freq_input = QLineEdit()
        self.freq_input.setPlaceholderText("Frequency MHz (e.g. 101.1)")
        self.freq_input.setFixedWidth(160)
        self.freq_input.returnPressed.connect(self.retune_from_text) 
        
        tune_button = QPushButton("Tune")
        tune_button.setFixedWidth(60)
        tune_button.clicked.connect(self.retune_from_text)

        # Station Dropdown
        self.station_combo = QComboBox()
        self.station_combo.setFixedWidth(260)
        for mhz, name in sorted(Local_stations.items()):
            self.station_combo.addItem(f"{mhz:.1f}  {name}", userData = mhz)
        # Preselect default freq 
        self.sync_combo_to_freq(self.current_Mhz)
        self.station_combo.currentIndexChanged.connect(self.retune_from_combo)
        
        # Pause Button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.setFixedWidth(70)
        self.pause_button.toggled.connect(self.toggle_pause)
        
        bar.addWidget(QLabel("Frequency:"))
        bar.addWidget(self.freq_input)
        bar.addWidget(tune_button)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Station:"))
        bar.addWidget(self.station_combo)
        bar.addStretch()
        bar.addWidget(self.pause_button)
        return bar
    
    #--------------------
    # Station Name Display 
    #--------------------
    def build_station_bar(self):
        frame = QFrame() 
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 4, 12, 4)
        
        self.station_name_label = QLabel(self.station_name(self.current_Mhz))
        self.freq_display_label = QLabel(f"{self.current_Mhz:.1f} MHz")
        
        layout.addWidget(self.station_name_label)
        layout.addStretch()
        layout.addWidget(self.freq_display_label)
        return frame
        
    #--------------------
    # RF Spectrum (IQ / SDR Rate)
    #--------------------
    def make_rf_spectrum(self):
        pw = pg.PlotWidget(title = "RF Spectrum (IQ Samples from SDR)")
        pw.setLabel('left', 'Power (dB)')
        pw.setLabel('bottom', 'Frequency Offset (Hz)')
        pw.setYRange(-80, 0)
        pw.showGrid(x = True, y = True, alpha = 0.12)
        
        freqs = np.fft.fftshift(np.fft.fftfreq(rf_fft_size, d = 1.0/sdr_rate))
        self.rf_freqs = freqs
        self.rf_curve = pw.plot(freqs, np.zeros(rf_fft_size))
        return pw
    
    #--------------------
    # Audio Frequency Spectrum and SNR
    #--------------------
    def make_audio_spectrum(self):
        pw = pg.PlotWidget(title = "Audio Spectrum and SNR")
        pw.setLabel('left', 'Power (dB)')
        pw.setLabel('bottom', 'Frequency (Hz)')
        pw.setXRange(0, audio_rate / 2, padding = 0)
        pw.setYRange(-81, 1, padding = 0)
        pw.enableAutoRange(x = False, y = False)
        pw.showGrid(x = True, y = True, alpha = 0.12)
        
        freqs = np.fft.rfftfreq(audio_fft_size, d = 1.0 / audio_rate)
        self.audio_spec_curve = pw.plot(
            freqs,
            np.full(len(freqs), -80.0)
        )
        
       # Shaded signal band (300 – 3400 Hz)
        sig_region = pg.LinearRegionItem(
            values=[300, 3400],
            brush=pg.mkBrush(46, 204, 113, 30),
            movable=False,
        )
        pw.addItem(sig_region)
        
        # SNR Text 
        self.snr_text = pg.TextItem(
            text = "SNR: -- dB",
            anchor = (1, 0),
        )
        self.snr_text.setPos(audio_rate / 2, 0)
        pw.addItem(self.snr_text)
        
        # Peak Marker Line 
        self.peak_line = pg.InfiniteLine(
            pos = 1000, angle = 90, 
            pen = pg.mkPen(width = 1, style = Qt.PenStyle.DashLine),
            label = "Peak",
            labelOpts = {"position": 0.85},           
        )
        pw.addItem(self.peak_line)
        
        self.audio_spec_pw = pw 
        return pw

    #--------------------
    # Waterfall
    #--------------------
    def make_waterfall(self):
        pw = pg.PlotWidget(title = "Waterfall (audio, time x freq)")
        pw.setLabel('bottom', 'Frequency (Hz)')
        pw.setLabel('left', 'Time')
        
        self.wf_img = pg.ImageItem() 
        pw.addItem(self.wf_img)
        self.wf_img.setColorMap(pg.colormap.get('inferno'))
        self.wf_img.setLevels((-80, -20))        
        return pw
    
    #--------------------
    # Time-Domain Waveform
    #--------------------
    def make_waveform(self):
        pw = pg.PlotWidget(title = "Audio Waveform (Time-Domain)")
        pw.setLabel('left', 'Amplitude')
        pw.setLabel('bottom', 'Sample')
        pw.showGrid(x = True, y = True, alpha = 0.12)
        pw.setXRange(0, audio_fft_size - 1, padding = 0)
        pw.setYRange(-1.5, 1.5, padding = 0)
        pw.enableAutoRange(x=False, y=False)        
        self.wave_curve = pw.plot(
            np.zeros(audio_fft_size)
        )
        
        return pw
    
    #--------------------
    # SDR Engine
    #--------------------
    def start_sdr(self):
        self.sdr_thread = SDRThread(FM_Audio.sdr)
        self.sdr_thread.error.connect(
            lambda m: self.status_label.setText(f"X {m}")
        )
        
        self.sdr_thread.start()
        self.status_label.setText("Streaming")
        
    #--------------------
    # Polling for Graphs
    #--------------------
    def poll(self):
        if self.paused:
            return 
        
        audio_updated = False 
        for _ in range(4):
            try:
                audio = self.audio_queue.get_nowait()
                n = min(len(audio), audio_fft_size)
                self.latest_audio = np.roll(self.latest_audio, -n)
                self.latest_audio[-n:] = audio[:n]
                audio_updated = True 
            except queue.Empty:
                break 
        
        if audio_updated:
            self.update_waveform()
            self.update_waterfall()
            self.update_audio_spectrum()
            self.update_rf_spectrum()
    
    #--------------------
    # Graph Updaters 
    #--------------------        
    def update_waveform(self):
        self.wave_curve.setData(self.latest_audio)
        
    def update_waterfall(self):
        # Audio fed waterfall
        spec = audio_fft(self.latest_audio)
        spec -= np.median(spec)  # Remove DC offset
        self.wf_buf.push(spec)
        self.wf_img.setImage(np.flipud(self.wf_buf.get()).T, autoLevels = False)
        
    def update_audio_spectrum(self):
        freqs = np.fft.rfftfreq(audio_fft_size, d = 1.0 / audio_rate)
        spec = audio_fft(self.latest_audio)
        self.audio_spec_curve.setData(freqs, spec)
        
        # SNR
        snr = estimate_snr(self.latest_audio)
        self.snr_history = np.roll(self.snr_history, -1)
        self.snr_history[-1] = snr
        self.snr_text.setText(f"SNR: {snr:.1f} dB")
        self.rssi_bar.setValue(int(snr))
        
        # Peak Marker
        pk_hz = np.clip(peak_freq(self.latest_audio), 0, audio_rate/2)
        self.peak_line.setValue(pk_hz)
    
    def update_rf_spectrum(self):
        iq = None 
        
        while True:
            try:
                iq = self.iq_queue.get_nowait() 
            except queue.Empty:
                break
        if iq is None:
            return

        spectrum = rf_fft(iq)
        self.rf_curve.setData(self.rf_freqs, spectrum)
        
    #--------------------
    # Frequency and Station Name Updater and Pause
    #--------------------  
    def retune_from_text(self):
        text = self.freq_input.text().strip()
        try:
            mhz = float(text)
            self.apply_retune(mhz)
            self.sync_combo_to_freq(mhz)
        except ValueError:
            self.status_label.setText("X Invalid Frequency X")
    
    def retune_from_combo(self, idx):
        mhz = self.station_combo.itemData(idx)
        if mhz is not None:
            self.apply_retune(mhz)
            self.freq_input.setText(f"{mhz:.1f}")
    
    def apply_retune(self, mhz):
        hz = mhz * 1e6
        FM_Audio.sdr.center_freq = hz
        FM_Audio.centerFreq = hz 
        self.current_Mhz = mhz 
        self.station_name_label.setText(self.station_name(mhz))
        self.freq_display_label.setText(f"{mhz:.1f}MhZ")
        self.status_label.setText(f"Tune to {mhz:.1f} Mhz")
        
    def sync_combo_to_freq(self, mhz):
        for i in range(self.station_combo.count()):
            if abs(self.station_combo.itemData(i) - mhz) < 0.05:
                self.station_combo.blockSignals(True)
                self.station_combo.setCurrentIndex(i)
                self.station_combo.blockSignals(False)
                return
            
    def station_name(self, mhz):
        for freq, name in Local_stations.items():
            if abs(freq - mhz) < 0.05:
                return name 
        return f"Unknown station at {mhz:.1f} MHz"
    
    def toggle_pause(self,checked):
        self.paused = checked 
        
        if checked:
            FM_Audio.stream.stop()
        else:
            FM_Audio.stream.start()
            
        self.pause_button.setText("Resume" if checked else "Pause")
        self.status_label.setText("Paused" if checked else "Streaming")
     
    #--------------------
    # Shutdown
    #--------------------     
    def closeEvent(self, event):
        self.timer.stop()
        self.sdr_thread.stop()
        self.sdr_thread.wait(3000)
        try: 
            FM_Audio.stream.stop()
            FM_Audio.stream.close() 
        except Exception:
            pass
        event.accept() 
    
#--------------------
# Entry Point
#--------------------     
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SDRVisualizer()
    win.show()
    sys.exit(app.exec())
            
            