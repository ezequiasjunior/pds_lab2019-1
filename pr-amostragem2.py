
#%%
#!/usr/bin/env python3
import numpy as np 
import matplotlib.pylab as plt
import scipy.io.wavfile as wf
import IPython.display as ipd
#%matplotlib

#%%
framerate = 44100
t = np.linspace(0,5,framerate*5)
data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
ipd.Audio(data,rate=framerate)

#%%
