#%%
import numpy as np
import scipy.sparse as sps

#%% datset
test = np.array([
[4, 5, 0, 5, 1, 0],
[0, 3, 4, 3, 1, 2],
[2, 0, 1, 3, 0, 4] ])

X = sps.csr_matrix(test)
print(X)
v = test.shape[0] #num of rows
c = test.shape[1]  # num of cols

#%% mean
(x, y, z) = scipy.sparse.find(test)
counts = np.bincount(x)
sums = np.bincount(x, weights=z)
np_sparse_avg = sums/counts

print(np_sparse_avg)

#%% corresponding
nnz_per_row = np.diff(X.indptr)
print(nnz_per_row)

#%%
Y = sps.csr_matrix((X.data - np.repeat(np_sparse_avg, nnz_per_row), X.indices, X.indptr),
                   shape=X.shape)
print(Y.todense())
print(type(Y))

#%%
A = np.squeeze(np.asarray(Y))



#%%
rows, cols = 3, 5
v = np.arange(0, rows)
print(v)
c = np.arange(cols, cols+cols)
print(c)
X = sps.rand(rows, cols, density=0.5, format='csr').toarray()
print(X)
#%%
lol = np.repeat(v, nnz_per_row)
Y = sps.csr_matrix((X.data * np.repeat(v, nnz_per_row), X.indices, X.indptr),
shape=X.shape)


#%%
nnz_per_row = np.diff(X.indptr)
Y = sps.csr_matrix((X.data - np.repeat(v, nnz_per_row), X.indices, X.indptr),shape=X.shape)
print(Y)
Y.data -= np.take(c, Y.indices)
print(Y)
#%%
x = X.toarray()
mask = x == 0
x *= v[:, np.newaxis]
x = c-x
x[mask] = 0
x


#%%
