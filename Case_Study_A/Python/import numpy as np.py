import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,np.pi/(2*np.sqrt(13)), 8)
dx = x[1]-x[0]
N = x.size

A = np.zeros((N,N))
b=np.zeros((N,0))
for n in range(1, N-1):
    A[n-1,n-1] = 1/dx**2
    A[n-1, n] = 13 - 2/dx**2
    A[n-1,n+1] = 1/dx**2
A[N-2,0] = 1
A[N-1,N-1] = 1
b[N-2]=1

plt.spy(A)
plt.show()
