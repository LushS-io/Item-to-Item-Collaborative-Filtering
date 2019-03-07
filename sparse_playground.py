#%% imports
import numpy as np
import pandas as pd
import scipy.sparse as sps

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

#%%
a = np.arange(12).reshape(3,4)
a = sps.csr_matrix(a)
print(a.todense())
print(a.T.todense())

#%% get dim
print(a[0].shape)
print(a[1].T.shape)

#%% play
x = a.tolil()
print(type(x))


#%% gah
a = np.arange(12).reshape(3, 4)
print(a)
[np.dot(a[i], a[i]) for i in range(3)]
np.einsum('ij,ij->i', a, a)

#%% get cos_sim
lawl = [(0,1),(0,2),(1,2)]
lawl = np.array(lawl)
print(type(lawl))

# print(np.dot(a[1],a[2].T))
print(np.dot(a,a.T)) # seeing if can only get by fancy index of known combos
#%% 
print(np.dot(a, a.T).diagonal())

#%%
a1 * a.T
a*a
a1.multiply(a1)

#%% multipy csr sparse with nd array element 3
