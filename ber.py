import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

# Parameters
N = int(1e6)  # number of bits or symbols
Eb_NO_dB = np.arange(-3, 60)  # multiple Eb/NO (SNR) values

# Transmitter
ip = np.random.rand(N) > 0.5  # generating 0,1 with equal probability
S = 2 * ip - 1  # BPSK modulation: 0 -> -1, 1 -> 1

# Simulation
nErr = np.zeros(len(Eb_NO_dB))

for i, Eb_NO in enumerate(Eb_NO_dB):
    n = np.sqrt(0.5) * (np.random.randn(N) + 1j * np.random.randn(N))  # white Gaussian noise
    h = np.sqrt(0.5) * (np.random.randn(N) + 1j * np.random.randn(N))  # Rayleigh fading channel
    y = h * S + np.sqrt(10 ** (-Eb_NO / 10)) * n  # received signal
    ipHat = (np.real(y / h) > 0).astype(int)  # receiver - hard decision decoding
    nErr[i] = np.sum(ip != ipHat)

# BER calculation
simBer = nErr / N
theoryBerAWGN = 0.5 * erfc(np.sqrt(10 ** (Eb_NO_dB / 10)))
theoryBer = 0.5 * (1 - np.sqrt(10 ** (Eb_NO_dB / 10) / (1 + 10 ** (Eb_NO_dB / 10))))

# Plot
plt.semilogy(Eb_NO_dB, theoryBerAWGN, 'cd-', linewidth=2)
plt.semilogy(Eb_NO_dB, theoryBer, 'bp-', linewidth=2)
plt.semilogy(Eb_NO_dB, simBer, 'mx-', linewidth=2)
plt.axis([-3, 35, 1e-5, 0.5])
plt.grid(True, which="both")
plt.legend(['AWGN-Theory', 'Rayleigh-Theory', 'Rayleigh-Simulation'])
plt.xlabel('Eb/No, dB')
plt.ylabel('Bit Error Rate')
plt.title('BER for BPSK modulation in Rayleigh channel')
plt.show()
