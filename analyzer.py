import numpy as np
import pyaudio

import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter


class SpectrumAnalyzer:
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 2048
    CHUNK = 512
    START = 0
    N = 512

    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []

    numtaps = 512
    lowcutoff = 250
    highcutoff = 350

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = self.FORMAT,
            channels = self.CHANNELS,
            rate = self.RATE,
            input = True,
            output = False,
            frames_per_buffer = self.CHUNK)
        # Main loop
        self.loop()

    def loop(self):
        try:
            while True :
                self.data = self.audioinput()
                #self.data = self.bandpass_filter()
                self.bandpass_filter_v2()
                self.fft()
                self.graphplot()

        except KeyboardInterrupt:
            self.pa.close()

        print("End...")

    def audioinput(self):
        ret = self.stream.read(self.CHUNK)
        ret = np.fromstring(ret, np.float32)
        return ret

    def fft(self):
        self.wave_x = range(self.START, self.START + self.N)
        self.wave_y = self.data[self.START:self.START + self.N]
        self.spec_x = np.fft.fftfreq(self.N, d = 1.0 / self.RATE)
        y = np.fft.fft(self.data[self.START:self.START + self.N])
        self.spec_y = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in y]

    def bandpass_filter(self):
        bandpass_coef = signal.firwin(self.numtaps, [self.lowcutoff, self.highcutoff], pass_zero=False, nyq=self.RATE * 2)
        bandpass_output = signal.convolve(self.data[self.START:self.START + self.N], bandpass_coef, mode='same')
        return bandpass_output


    def bandpass_filter_v2(self):
        b, a = self.design_filter()
        self.data = lfilter(b, a, self.data[self.START:self.START + self.N])


    def design_filter(self, order=5):
        nyq = 0.5 * self.RATE
        low = self.lowcutoff / nyq
        high = self.highcutoff / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def graphplot(self):
        plt.clf()
        # wave
        plt.subplot(311)
        plt.plot(self.wave_x, self.wave_y)
        plt.axis([self.START, self.START + self.N, -0.5, 0.5])
        plt.xlabel("time [sample]")
        plt.ylabel("amplitude")
        #Spectrum
        plt.subplot(312)
        plt.plot(self.spec_x, self.spec_y, marker= 'o', linestyle='-')
        plt.axis([0, self.RATE / 2, 0, 50])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        #Pause
        plt.pause(.01)

if __name__ == "__main__":
    spec = SpectrumAnalyzer()