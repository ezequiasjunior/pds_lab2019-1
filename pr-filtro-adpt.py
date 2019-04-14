"""
Como plotar, inicia kernel, plota um default
mantém o kernel...
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt

a = np.random.rand(30)
b = np.random.randn(10, 3)*10
np.savez('test_file', nome0=a, nome1=b)


e = np.load('test_file.npz')
print('Dados armazenados em:\n{}'.format(e.files))

x = e['nome0']
y = e['nome1']

np.set_printoptions(4)
print(x)
#%% plot
with plt.style.context('ggplot', True):
    fig, ax = plt.subplots()
    ax.plot(y, 'o-')
    plt.savefig('test_fig.png')
    plt.show()

#%%
vt_snr = [1,2,3]
users = 9
ntx, mrx = 3,5
ini = 'opt'
scenario = 'trolololo'
func = 'wassd'
max_iterations = 333
epsilion = 0.0000003
realizations = 1e4

print(f'''
Initializing simulation for the current values of SNR:
\n{vt_snr}\n
Number of users: {users}
TX antennas, RX antennas: {ntx}, {mrx}
Type of vk and gk initialization: {ini}
Scenario: {scenario}
Algorithm: {func}
Iterations: {max_iterations}
Tolerance: {epsilion}
Number of channel realizations: {realizations}''')