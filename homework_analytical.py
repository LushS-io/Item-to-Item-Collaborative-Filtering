# Homework 2 analytical for cs315
# %% Import
import pandas as pd
import numpy as np
import math as mth
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from dfply import *
from itertools import starmap
import scipy as scipy
import time as time
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import itertools as iter
import sklearn.preprocessing as pp

# %% Methods
"""Takes 2 vectors a, b and returns the cosine similarity according to the definition of the dot product"""
def cosine_similar(mat):
    row_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return row_normed_mat.T * row_normed_mat

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def combs_nd(a, r, axis=0):
    a = np.asarray(a)
    if axis < 0:
        axis += a.ndim
    indices = np.arange(a.shape[axis])
    dt = np.dtype([('', np.intp)]*r)
    indices = np.fromiter(combinations(indices, r), dt)
    indices = indices.view(np.intp).reshape(-1, r)
    return np.take(a, indices, axis=axis)


def import_hw2data_to_pd_df():
    # dataset from hw 2
    user_1 = np.array([1,0,3,0,0,5,0,0,5,0,4,0])
    user_2 = np.array([0,0,5,4,0,0,4,0,0,2,1,3])
    user_3 = np.array([2,4,0,1,2,0,3,0,4,3,5,0])
    user_4 = np.array([0,2,4,0,5,0,0,4,0,0,2,0])
    user_5 = np.array([0,0,4,3,4,2,0,0,0,0,2,5])
    user_6 = np.array([1,0,3,0,3,0,0,2,0,0,4,0])

    test_1 = np.array([4, 5, 0, 5, 1, 0])
    test_2 = np.array([0, 3, 4, 3, 1, 2])
    test_3 = np.array([2, 0, 1, 3, 0, 4])

    # print combination of similarities
    print(cos_sim(user_1, user_2))
    print(cos_sim(user_1, user_3))
    print(cos_sim(user_2, user_3))
    # %% Create pd_df from array data
    '''Append 3 users into df'''
    # users = pd.DataFrame(data=(user_1, user_2, user_3,user_4,user_5,user_6))
    users = pd.DataFrame(data=(test_1, test_2, test_3))
    return users


# %% Create pandas and numpy df
users = import_hw2data_to_pd_df()
df_users = users
np_users = users.to_numpy()

# %% Create CSR Sparse Matrix
np_sprase = scipy.sparse.csr_matrix(np_users)
print('before sparse\n{}'.format(users))
print(np_sprase)

# %% create csc 
csc_mat = scipy.sparse.csc_matrix(np_users)
plp=cosine_similar(csc_mat)
print(plp.todense())

# %% get mean of each row
(x, y, z) = scipy.sparse.find(np_sprase)
counts = np.bincount(x)
sums = np.bincount(x, weights=z)
np_sparse_avg = sums/counts

'''check avg'''
print(np_sparse_avg)
'''check type'''
print(type(np_sparse_avg))

#%% normalize matrix
X = np_sprase
nnz_per_row = np.diff(X.indptr)

print(nnz_per_row)

Y = sps.csr_matrix((X.data - np.repeat(np_sparse_avg, nnz_per_row), X.indices, X.indptr),
                   shape=X.shape)
print(Y.todense())
print()
print(Y.T.todense())

#%% play
yao = cosine_similar(Y)
print(yao.todense())

#%% stacktest
A  =  Y

start = time.time()

## base similarity matrix (all dot products)
## replace this with A.dot(A.T).toarray() for sparse representation

# similarity = A.dot(A.T)
similarity = A.dot(A.T).toarray()

print(similarity)

## squared magnitude of preference vectors (number of occurrences)

# square_mag = np.diag(similarity)
square_mag = similarity.diagonal()
# square_mag = similarity.todia()

print(square_mag)
print()

end = time.time()
## inverse squared magnitude
inv_square_mag = 1 / square_mag
# inv_square_mag = sps.linalg.inv(square_mag)

print(inv_square_mag)

## if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
inv_square_mag[np.isinf(inv_square_mag)] = 0

## inverse of the magnitude
inv_mag = np.sqrt(inv_square_mag)
# inv_mag = inv_square_mag.sqrt()

## cosine similarity (elementwise multiply by inverse magnitudes)
cosine = similarity * inv_mag
cosine = cosine.T * inv_mag

print(cosine)
print(end-start)

#%% another GAH
print(Y * Y.T)

#%% section D 
yah = Y.todense()
print(yah)
print(cos_sim(yah[0],yah[1].T))
print(cos_sim(yah[0], yah[2].T))
print(cos_sim(yah[1],yah[2].T))
print()
print(yah[0])
print(yah[1].T)

#%% 
dot_prod = np.dot(Y,Y.T)
print(dot_prod.todense())



#%% compare

cosine_similarity(df_users)

#%% csr to coo matrix for fast iterations

cx = scipy.sparse.coo_matrix(Y)

for i, j, v in zip(cx.row, cx.col, cx.data):
    print ("(%d, %d), %s" % (i, j, v))


#%%
# A = np.squeeze(np.asarray(x))
A = Y.todense()
combos = combs_nd(A,2)
print(combos)
for pair in combos:
     x1 = pair[0]
     x2 = pair[1]
     print(cos_sim(x1, x2))

#compare
cosine_similarity(A)
