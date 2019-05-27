#%% [markdown]
# ### Prática - Filtragem Espacial
# 
# cuidado na matriz A ela é N elementos x M ondas/fontes, eu pensei em M fontes por N elementos
# 
# A não varia no tempo considerando uma coerencia... no intervalo de tempo T a informação angular é a mesma
# 
# R = R sinal + R noise => EVD.... Us = subspaço do sinal | Un subspaço do ruído -> ortogonais
# 
# - o que é a matriz covariância
# 
# Us.H @ Un = 0
# 
# R amostral =  Maxima verossimilhança média de x@xH 
# 
# vetor de apontamento média do a($\theta$)
# 
# #### métodos de busca
# 
# correlação aH(x) @ a(x0) max w/ x = x0
# 
# cor(x) = aH(x) @ x(t) ... varrer com um x de um em um grau muda o gráfico...mais fino e variar o numero de antenas?
# X é uma unica amostra.... será que varia de t ???
# 
# M maiores picos representam as estimativas,  quantos maiores que um treshhold 
# acima de 45%
# 
# para o metodo do BF Convencional
# 
# formula do wBF slides formula que varia com theta
# matriz R dependendo do ""buffer que armazena T"
# X considera todas as amostras de T pq ele olha pra matriz R desempenho melhor
# 
# 
# Para o Music utiliza a SVD de R na fórmula da potencia em funlçao de theta 
#(subspaço do ruido)
# picos do espectro explorando o conceito da ortogonalidade do sinal e do ruído 
# 
# achar uv.... N primeiros valores estão no sub do sinal, os M-N  
# N primeiras colunas é sinal o resto é ruido \[coluna| ... | coluna\]
# 
# saber o numero de maiores autovalores .... e/emax dá o numero de angulos "" 
#por cima"" ordenados....
# 
# ESPRIT solução fechada  
# entra pseudo inversa us\[0:size-1\] e us\[1:size\]  
# usa argumento numero complexo ( ver função pois retorna radiano)  
# 
# angulos diferentes Us rank completo
# 
# ---------------
# ---------------
# Erro entre os métodos o angulo que ele cai 
# 
# se for percorrer a estimativa a cada snr precisa fazer um monte carlo média 
# dos valores
# 
# para os plots lembrar de normalizar pra bater certinho
# 
# Pratica vai variar o N, snapshot = T
# pegar os 3 maiores picos , calcula o max, elimina, calcula o max, elimina, 
#calcula o max = 3 máx
# 
# fazer um pandas pra complexidade ????
# 
# 10 db pra varia r o T... será que dá pra fazer uma superficie??
# 
# plot pelo numero de sensores.... 5 a 50
# 
# ver como fica o arranjo... e após a estimativa
# 
# 
# para a uRA.... lembrar do vec... percorrer por matrizes se der tempo....
# 
# pra comparativo colocar só os pandas e falar...

#%%
#!/usr/bin/env python3
import numpy as np 
from numpy.random import rand, randn
import matplotlib.pylab as plt
from numpy.linalg import eig, pinv
from scipy.signal import find_peaks

# calcular o tempo necessário de busca ? seria o time it pra comprarar
# comparação justa todos os metods no mesmo realiz

#%%
# Funções:
def ula(N, d, theta):
    dx = np.arange(N)*d  # Indices dos espaçamentos "(nx-1)*dx"
    # Resposta do arranjo linear:
    return np.exp(-1j*2*np.pi*dx*np.sin(theta))

def get_angles(x, angles):
    # Extração dos picos dos espectros espaciais:
    idx = find_peaks(x.reshape(x.size,), height=0.45)
    return np.rad2deg(angles[idx[0]])

# Método da Correlação:
def corr_bf(X, angles):
    corr = np.zeros((angles.size, 1), dtype=complex)
    for idx, a in enumerate(angles):
        corr[idx] = np.conj(ula(N, d, a)).T@X
    # Normalização:
    corr = np.abs(corr)
    return corr/(corr.max())

# Método Beamformer Convencional:
def conv_bf(X, angles):
    # Matriz de covariância estimada:
    mt_R = np.zeros((X.shape[1], X.shape[1]), dtype=complex)
    for t in range(X.shape[0]):
        mt_R += (X[t]@np.conj(X[t]).T)

    # Cálculo da matriz de covariância estimada:
    mt_R /= X.shape[0]
    # Espectro espacial:
    Pbf = np.zeros((angles.size, 1), dtype=complex)
    for idx, a in enumerate(angles):
        st_vec = ula(N, d, a)
        Pbf[idx] = np.conj(st_vec).T@mt_R@st_vec/\
                   np.sqrt(np.conj(st_vec).T@st_vec)
    # Normalização:
    Pbf = np.abs(Pbf)
    return Pbf/(Pbf.max())

# Método MUSIC
def music_bf(X, angles, m):
    # Matriz de covariância estimada:
    mt_R = np.zeros((X.shape[1], X.shape[1]), dtype=complex)
    for t in range(X.shape[0]):
        mt_R += (X[t]@np.conj(X[t]).T)

    # Cálculo da matriz de covariância estimada:
    mt_R /= X.shape[0]
    # Decomposição em autovalores de R:
    evalue, evec = eig(mt_R)
    Uv = evec[:, m:]    # Subespaço do ruído
    # Espectro espacial:
    Pmsc = np.zeros((angles.size, 1), dtype=complex)
    for idx, a in enumerate(angles):
        st_vec = ula(N, d, a)
        Pmsc[idx] = np.conj(st_vec).T@st_vec/\
                np.sqrt(np.conj(st_vec).T@Uv@np.conj(Uv).T@st_vec)
    # Normalização:    
    Pmsc = np.abs(Pmsc)
    return Pmsc/(Pmsc.max())

# Método ESPRIT:
def esprt_bf(X, m, d):
    # Matriz de covariância estimada:
    mt_R = np.zeros((X.shape[1], X.shape[1]), dtype=complex)
    for t in range(X.shape[0]):
        mt_R += (X[t]@np.conj(X[t]).T)

    # Cálculo da matriz de covariância estimada:
    mt_R /= X.shape[0]
    # Decomposição em autovalores de R:
    evalue, evec = eig(mt_R)
    # Estimação:
    Us = evec[:, :-(X.shape[1]-m)]    # Subespaço do sinal
    # mt_psi = pinv(Us[:-1, :]) @ Us[1:, :]
    mt_psi = pinv(Us[1:, :]) @ Us[:-1, :]
    val, vec = eig(mt_psi)
    angles = np.angle(val)
    # Retornando ângulos estimados:
    return np.rad2deg(np.sort(np.arcsin(angles/(2*np.pi*d))))


#%%
# Parâmetros:
snapshot = 500              # Snapshots
M = 3                       # Número de fontes de sinal
N = 10                      # Número de sensores do arranjo
d = 0.5                     # Espaçamento (lambda = 1)        
delta = np.radians(0.01)    # Passo para os ângulos
angles = np.arange(0, np.pi/2+delta, delta)    # Ângulos de Varredura [0,2pi]
snr = 10                    # Relação sinal-ruído [dB]
# Angulos de elevação, assumindo que o arranjo esteja no eixo x:
theta = np.array([20, 35, 65])
theta_r = np.radians(theta)
# Construção da matriz de resposta do arranjo:
mt_A = np.zeros((N,M), dtype=complex)
for idx, val in enumerate(theta_r):
    mt_A[:, idx] = ula(N, d, val)

# Modelo de sinal:
mt_X = np.zeros((snapshot, N, 1), dtype=complex)
mt_R = np.zeros((N, N), dtype=complex)
# Para cada snapshot:
for t in range(snapshot):
    # Sinal de cada fonte:
    signal = rand(M, 1) + 1j*rand(M, 1)
    # Ruído de cada sensor:
    vt_n = (randn(N, 1) + 1j*randn(N, 1))/np.sqrt(10**(0.1*snr))
    # Sinal recebido no instante t:
    mt_X[t] = mt_A@signal + vt_n

#%%
# Método da Correlação:
# Cálculo da média dos dt primeiros snapshots:
dt = 5
c = np.zeros((dt, angles.size, 1))
for i in range(dt):
    c[i] = corr_bf(mt_X[i], angles)

corr = np.mean(c, axis=0)

#%%
# Método Beamformer Convencional:
beamf = conv_bf(mt_X, angles)

#%%
# Método MUSIC
music = music_bf(mt_X, angles, M)

#%%
# Método ESPRIT:
esprit = esprt_bf(mt_X, M, d)


#%%
a = np.rad2deg(angles)
with plt.style.context('ggplot', True):
    plt.figure()
    plt.title('Comparativo - Métodos de Estimação de AoA')
    
    plt.plot(a, corr, '-.', label='BF Corr.')
    plt.plot(a, beamf, '--', label='BF Conv.')
    plt.plot(a, music, '-', label='MUSIC')

    plt.plot(esprit, 0.75*(np.ones(M)), linestyle='none', marker='o', 
                                        color='r', 
                                        markersize=3, fillstyle='full', 
                                        label='ESPRIT')

    plt.plot(theta, 0.75*(np.ones(M)), linestyle='none', color='b', 
                                        marker='o', 
                                        markersize=8, fillstyle='none', 
                                        label='$\Theta$ Exato')
    # plt.grid()
    for value in theta:
        plt.plot(value*np.ones(100), np.linspace(0,1,100), 'k:', 
                 label=f'$\Theta$ = {value}°')
    # plt.xticks(np.arange(0, 91, 5))
    plt.axis([0, 90, 0, 1.01])
    plt.legend(loc='best')
    plt.show()

#%%
# ii=find_peaks((abs(beamf)/max(abs(beamf))).reshape(angles.size,), height=0.45)
# jj=find_peaks((abs(music)/max(abs(music))).reshape(angles.size,), height=0.45)
# kk=find_peaks((abs(corr)/max(abs(corr))).reshape(angles.size,), height=0.45)

# print(f'''
# {3}
# {np.rad2deg(angles[kk[0]])}
# {np.rad2deg(angles[ii[0]])}
# {np.rad2deg(angles[jj[0]])}
# {np.rad2deg(esprit)}
# ''')
# np.arange(5).max()
