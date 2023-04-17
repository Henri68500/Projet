import numpy as np
import matplotlib.pyplot as plt

N = 20 #Nombre de points du maillage
Xat = 0.50 #Position de l'atome dans la cellule unité
dx = 1/N #Pas du maillage
Vmax = 10000 #Potentiel maximal (pour éviter les divergences)
nb_k = 40 #Nombre de valeurs que prend k

def V(x):
    if abs(x-Xat) <= 0.00001:
        return Vmax
    else:
        return min(Vmax,1/abs(x-Xat))

V = np.vectorize(V)

maillage = np.array([j*dx for j in range(N+1)])
liste_k = np.linspace(0.0001,2*np.pi-0.0001,nb_k)

Pot_discret = V(maillage)

def make_matrix(u,v,w,N):
    M = v*np.identity(N)
    for i in range(N-1):
        M[i+1,i]=u
        M[i,i+1]=w
    return M

A = make_matrix(1,-2,1,N+1)
B = make_matrix(0,-1,1,N+1)

A[0,N]=1
A[N,0]=1
B[N,0]=1

trac = []

for k in liste_k:
    C_k = np.zeros((N+1,N+1))
    for i in range(N+1):
        C_k[i,i]=V(maillage[i])+k**2

    H_k = (1/dx**2) * A - (2*complex(0,1)*k/dx) * B + C_k

    val_propres = np.linalg.eig(H_k)[0]

    trac.append(val_propres)


trac = np.transpose(np.array(trac))

print(trac)

for j in range(N):
    plt.plot(liste_k,trac[j])

plt.show()