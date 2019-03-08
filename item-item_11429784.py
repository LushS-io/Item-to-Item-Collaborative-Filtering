''' Homework 2 programming portion '''
#%% imports
import pandas as pd
import numpy as np
import math as mth
# from dfply import *
# from itertools import starmap
import scipy as scipy
import time as time
from scipy.sparse import csr_matrix
import scipy.sparse as sps
# import itertools as iter

#testing purposes
# from scipy.spatial.distance import cosine
# from sklearn.metrics.pairwise import cosine_similarity

#%% import datasets
links = pd.read_csv("./movie-lens-data/links.csv")
movies = pd.read_csv("./movie-lens-data/movies.csv")
ratings = pd.read_csv("./movie-lens-data/ratings.csv")
tags = pd.read_csv("./movie-lens-data/tags.csv")

#%% list attributes
print(list(movies.columns.values)) #another way to list
print(list(ratings))
print(list(links))
print(list(tags))


#%% join movies and ratings
# movies.join(other=ratings,on='movieId',lsuffix="_movies",rsuffix="_ratings")
df = pd.merge(movies,ratings, on='movieId', how='outer')

#%% let's see merged df
list(df)

#%% drop all features besides rating
df_mov_rat = ratings[['rating','movieId']].copy()
print(df_mov_rat.shape) #check shape
print(type(df_mov_rat)) #check type

#%% create np array
np_mov_rat = df_mov_rat.to_numpy() #create np array

#%% gass
cosine_similarity(np_mov_rat)

# %% Create CSR Sparse Matrix
# csr_mat = scipy.sparse.csr_matrix(np_mov_rat)

# %% Create small CSR Sparse Matrix for testing
csr_mat = scipy.sparse.csr_matrix(np_mov_rat[:100,])


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
print(Run_time) #test runtime





#%% get cosine similarity
start_time = time.time()
A = Y # carry over normalized matrix to get cosine_similarity

# base similarity matrix (all dot products)
# replace this with A.dot(A.T).toarray() for sparse representation

# similarity = A.dot(A.T)
similarity = A.dot(A.T).toarray()


# squared magnitude of preference vectors (number of occurrences)
# square_mag = np.diag(similarity)
square_mag = similarity.diagonal()

# inverse squared magnitude
inv_square_mag = 1 / square_mag
# inv_square_mag = sps.linalg.inv(square_mag)

# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
inv_square_mag[np.isinf(inv_square_mag)] = 0

# inverse of the magnitude
inv_mag = np.sqrt(inv_square_mag)
# inv_mag = inv_square_mag.sqrt()

# cosine similarity (elementwise multiply by inverse magnitudes)
cosine = similarity * inv_mag
cosine = cosine.T * inv_mag

end_time = time.time()

print(cosine)


Run_time = end_time - start_time
print(Run_time)
#%%
