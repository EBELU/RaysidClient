import time
import numpy as np
from .data_classes import SpectrumResult

class SpectrumAccumulator:
    def __init__(self, n_channels=1800):
        self.n = n_channels
        self.reset()

    def reset(self):
        self.spectrum = np.zeros(self.n, dtype=np.float64)

    def insert(self, start_ch, end_ch, spectrum_chunk):
        
        mask = spectrum_chunk > ((1e3 + np.max(self.spectrum)) * 5)
        spectrum_chunk = np.where(mask, self.spectrum, spectrum_chunk)
        
        if not ((start_ch == 0) and (end_ch == 0)):
            self.spectrum[start_ch:end_ch] = spectrum_chunk[start_ch:end_ch]
        

    def snapshot(self):
        return SpectrumResult(
            spectrum=self.spectrum.copy(),
            timestamp=time.time()
        )