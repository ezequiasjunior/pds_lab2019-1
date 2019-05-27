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
# Pratica não vai variar o N, snapshot = T
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
from numpy.linalg import eig
# correlação tbm....

'''
snr[lin] =  10^(0.1*snr)
para a snr tem o ruído do sinal como inverso... snr =1/sigma^2 , simga = 1/sqrt
#(snr[lin])

fazer funções para criação de arranjos -> parametro (n, theta)

plots de corr x angulos, fazer uma reta para eixo x = angulo desejado
meu angulos tao em radiano no plot mudar os valores para deg

plots de pot do BF pelo angulo

calcular o tempo necessário de busca ? seria o time it pra comprarar
'''
#%%
def ula(N, d, theta):
    dx = np.arange(N)*d  # Indices dos espaçamentos "(nx-1)*dx"
    # Resposta do arranjo linear:
    return np.exp(-1j*2*np.pi*dx*np.sin(theta))

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

# #------------------------------------------------------------------------------
#     # Matriz de covariância estimada:
#     mt_R += (mt_X[t]@np.conj(mt_X[t]).T)

# # Cálculo da matriz de covariância estimada:
# mt_R /= snapshot
#-----------------------------------------------------------------------------
# # Decomposição em autovalores de R:
# evalue, evec = eig(mt_R)

#%%
# Método da Correlação:
def corr_bf(X, angles):
    corr = np.zeros((angles.size, 1), dtype=complex)
    for idx, a in enumerate(angles):
        corr[idx] = np.conj(ula(N, d, a)).T@X
    return corr

# Cálculo da média dos dt primeiros snapshots:
dt = 5
c = np.zeros((dt, angles.size, 1), dtype=complex)
for i in range(dt):
    c[i] = corr_bf(mt_X[i], angles)

corr = np.mean(c, axis=0)

#%%
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
        Pbf[idx] = np.conj(ula(N, d, a)).T@mt_R@ula(N, d, a)/\
                   np.sqrt(np.conj(ula(N, d, a)).T@ula(N, d, a))
    
    return Pbf

beamf = conv_bf(mt_X, angles)

#%%
# Método MUSIC
def music_bf(X, angles):
    # Matriz de covariância estimada:
    mt_R = np.zeros((X.shape[1], X.shape[1]), dtype=complex)
    for t in range(X.shape[0]):
        mt_R += (X[t]@np.conj(X[t]).T)

    # Cálculo da matriz de covariância estimada:
    mt_R /= X.shape[0]
    # Decomposição em autovalores de R:
    evalue, evec = eig(mt_R)
    

    return

#%%
# mudar x ticks
plt.figure()
# plt.grid()
plt.plot(np.rad2deg(angles), np.abs(corr)/np.max(abs(corr)))
for value in theta:
    plt.plot(value*np.ones(100),np.linspace(0,1,100),'--')
plt.show()
# FAZER FUNÇÃO PARA EXTRAIR O m MAIORES VALORES..
# comparação justa todos os metods no mesmo realiz

#%%
# mudar x ticks
plt.figure()
# plt.grid()
plt.plot(np.rad2deg(angles), np.abs(beamf)/np.max(abs(beamf)))
for value in theta:
    plt.plot(value*np.ones(100),np.linspace(0,1,100),'--')
plt.show()
#%%
np.set_printoptions(2)
print(f'''
Cov:
{mt_A}

{evec.shape}
{angles.shape}
''')
