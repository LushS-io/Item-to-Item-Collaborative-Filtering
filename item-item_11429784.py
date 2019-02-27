''' Homework 2 programming portion '''
#%% imports
import pandas as pd
import numpy as np
import math as mth
from itertools import combinations
from dfply import *
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

#%% function for cosine similarity
"""Takes 2 vectors a, b and returns the cosine similarity according to the definition of the dot product"""
def cos_sim(a, b):
    dot_product = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

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

#%% we only need rating and movie id for first part
df_mov_rat = ratings[['rating','movieId']].copy()
df_mov_rat.shape
type(df_mov_rat )

#%% cut down to sample dataset put into nparray since faster
np_small = df_mov_rat.iloc[0:10,:]
np_small = np_small.to_numpy()
# type(np_small)
np_small.shape
#%% check
print(np_small)
#%% run
for row in np.nditer(np_small):
    print(row)
#%% to check algorithm run pairwise cosine from sklearn
check = cosine_similarity(np_small)
print(check)
#%%
print(np_small['rating'])
list(np_small)
#%%
for a in np.nditer(x):
    print(a)

#%%

# group_by movie id -> then put ratings for group into a list
# run cosine similarity on combinations based on movie_id