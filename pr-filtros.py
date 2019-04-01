
#%%
#!/usr/bin/env python3
import numpy as np 
import matplotlib.pylab as plt
import scipy.io.wavfile as wf
from scipy import signal
from scipy.fftpack import fft, fftfreq
import IPython.display as ipd
%matplotlib
plt.style.use('seaborn')

#%%
rate, wavefile = wf.read('audio-pr-filter.wav')
print(rate, wavefile.shape)

audioL = wavefile[:, 0]

# Construindo vetor de tempo para representação do sinal:
time = np.linspace(0, audioL.size/rate, audioL.size)
#%%
# Reprodução:
print('Audio:')
ipd.Audio(data=audioL, rate=rate)

#%%
# Visualização do sinal
plt.figure('audio')
plt.plot(time, audioL)
plt.show()

#%%
# Visualização do espectro:
# Frequências e fft: audio 1
freq = fftfreq(audioL.size) * rate
Y = 2*np.abs(fft(audioL))/audioL.size
plt.figure()
plt.plot(freq, Y, 'C0-')
plt.xlim(0, rate/2)
plt.show()

# Psd do sinal de audio 1:
f, P_den = signal.welch(audioL, rate, nperseg=1024)
plt.figure()
plt.semilogy(f, P_den, 'C0-')
plt.xlim(0, rate/2)
plt.show()

#%%
