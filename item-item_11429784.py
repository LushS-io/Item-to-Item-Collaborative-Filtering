''' Homework 2 programming portion '''
#%% imports
import pandas as pd
import numpy as np
import math as mth
from itertools import combinations
from dfply import *
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

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
#%%

#%%
