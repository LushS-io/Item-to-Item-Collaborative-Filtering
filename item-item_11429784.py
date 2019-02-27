''' Homework 2 programming portion '''
#%% imports
import pandas as pd
import numpy as np
import math as mth
from itertools import combinations
from dfply import *
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
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

#%%
list(df)

#%%
# group_by movie id -> then put ratings for group into a list
# run cosine similarity on combinations based on movie_id
new = ratings[['rating','movieId']].copy()
new = new[:100]
new.shape
#%%
dist_out = 1-pairwise_distances(new, metric="cosine")
test = cos_sim(new[0],new[1])
#%%
dist_out
#%%
