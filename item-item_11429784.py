''' Homework 2 programming portion '''
# %% imports
import pandas as pd
import numpy as np
import math as mth
# from dfply import *
# from itertools import starmap
import scipy as scipy
import time as time
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import itertools as iter
# python3 has zip already

# testing purposes
# from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

#%% helper func.


def adj_to_neighbor_dict(adj):
    assert hasattr(adj, "__iter__")

    neighbor_dict = collections.defaultdict(lambda: set())
    for i, j in adj:
        if i == j:
            continue
        neighbor_dict[i].add(j)
        neighbor_dict[j].add(i)
    return neighbor_dict


def get_neighbors_2d(npmatrix):
    assert len(npmatrix.shape) == 2
    I, J = range(npmatrix.shape[0]-1), range(npmatrix.shape[1]-1)
    adj_set = set(
        (npmatrix[i, j], npmatrix[i+1, j])
        for i in I
        for j in J
    ) | set(
        (npmatrix[i, j], npmatrix[i, j+1])
        for i in I
        for j in J
    )
    return adj_to_neighbor_dict(adj_set)


# %% another test dataset
user_1 = np.array([1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0])
user_2 = np.array([0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3])
user_3 = np.array([2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0])
user_4 = np.array([0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0])
user_5 = np.array([0, 0, 4, 3, 4, 2, 0, 0, 0, 0, 2, 5])
user_6 = np.array([1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0])

pd_df = pd.DataFrame(data=(user_1,user_2,user_3,user_4,user_5,user_6))
np_mov_rat = sps.csr_matrix(pd_df)
# %% test dataset
# test_1 = np.array([4, 5, 0, 5, 1, 0])
# test_2 = np.array([0, 3, 4, 3, 1, 2])
# test_3 = np.array([2, 0, 1, 3, 0, 4])
# pd_df = pd.DataFrame(data=(test_1, test_2, test_3))
# np_mov_rat = sps.csr_matrix(pd_df)

# %% import datasets
links = pd.read_csv("./movie-lens-data/links.csv")
movies = pd.read_csv("./movie-lens-data/movies.csv")
ratings = pd.read_csv("./movie-lens-data/ratings.csv")
tags = pd.read_csv("./movie-lens-data/tags.csv")

# %% list attributes
print(list(movies.columns.values))  # another way to list
print(list(ratings))
print(list(links))
print(list(tags))


# %% join movies and ratings
# movies.join(other=ratings,on='movieId',lsuffix="_movies",rsuffix="_ratings")
df = pd.merge(movies, ratings, on='movieId', how='outer')

# %% let's see merged df
list(df)
# %% drop all features besides rating
df_mov_rat = df[['rating', 'movieId', 'userId']].copy()
print(df_mov_rat.shape)  # check shape
print(type(df_mov_rat))  # check type
df_mov_rat.head()
''' looks like the profile is not in correct format '''

# %% reshape df into proper matrix
df = df_mov_rat
df.head()

# df = df.reset_index().dropna() # drop na

df = df.pivot(index='userId',columns='movieId',values='rating')
df = df.fillna(0) # fill nan with 0
print(df)

# %% ---------------Create CSR Sparse Matrix *** -----------------------------------
# csr_mat = scipy.sparse.csr_matrix(df) ### test SWTICH ***
csr_mat = sps.csr_matrix(np_mov_rat)
# %% Create small CSR Sparse Matrix for testing
# csr_mat = scipy.sparse.csr_matrix(np_mov_rat[:5000, ])


# %% get mean of each row
(x, y, z) = scipy.sparse.find(csr_mat)
counts = np.bincount(x)
sums = np.bincount(x, weights=z)
csr_avg = sums/counts

# %% normalize
start_time = time.time()
X = csr_mat
nnz_per_row = np.diff(X.indptr)

# print(nnz_per_row)

Y = sps.csr_matrix((X.data - np.repeat(csr_avg, nnz_per_row), X.indices, X.indptr),
                   shape=X.shape)
# print(Y.todense()) # check normalize mat
# print()
# print(Y.T.todense()) #check normalize mat transposed
end_time = time.time()
Run_time = end_time - start_time
print(Run_time)  # test runtime


# %%
csr_mat.shape
# print(Y.todense())
# %% ----------- get cosine similarity --------

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=1000)

start_time = time.time()
A = Y  # carry over normalized matrix to get cosine_similarity

# base similarity matrix (all dot products)
# replace this with A.dot(A.T).toarray() for sparse representation

# similarity = A.dot(A.T)
similarity = A.dot(A.T).toarray()

#x print(similarity)

# squared magnitude of preference vectors (number of occurrences)
# square_mag = np.diag(similarity)
square_mag = similarity.diagonal()

# inverse squared magnitude
inv_square_mag = 1 / square_mag
# inv_square_mag = sps.linalg.inv(square_mag)
#x print(inv_square_mag)
# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
inv_square_mag[np.isinf(inv_square_mag)] = 0
#x print(inv_square_mag)

# inverse of the magnitude
inv_mag = np.sqrt(inv_square_mag)
# inv_mag = inv_square_mag.sqrt()
#x print(inv_mag)
# cosine similarity (elementwise multiply by inverse magnitudes)
cosine = similarity * inv_mag
cosine = cosine.T * inv_mag

end_time = time.time()

# print(np.isfinite(cosine.T).all()) # check if 
cosine_mat = cosine.T
print(cosine_mat)


Run_time = end_time - start_time
print(Run_time)
# %% for comparison purpose ;)
start = time.time()

test = cosine_similarity(Y)
print(test)


end = time.time()
total = end - start
print(total)

# %% to coo 
coo_cos = sps.coo_matrix(cosine_mat)
print(coo_cos)

# %% get nearest neighbors
knn = 5  # set the num of nearest neighbors to find

def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[0], x[2]))

t = sort_coo(coo_cos)
t = sps.csr_matrix(t)
print(t.todense())

gah = pd.DataFrame(t.todense())
print(gah)
# the idea
# sort and get the top 5
# run weighted average on those top 5

# %%
NearestNeighbors(n_neighbors=5,)





#%%
