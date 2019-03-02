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

#%% drop all features besides rating
df_mov_rat = ratings[['rating','movieId']].copy()
print(df_mov_rat.shape) #check shape
print(type(df_mov_rat)) #check type

np_mov_rat = df_mov_rat.to_numpy() #create np array

#%% Create small subset to test on 
df_np_small = df_mov_rat.iloc[0:10,:]
np_small = df_np_small.to_numpy()
# type(np_small)
np_small.shape

#%% Print to check test df 
print(np_small)
print(df_np_small)
#%% run
for row in np.nditer(np_small):
    print(row)
#%% to check algorithm run pairwise cosine from sklearn
check = cosine_similarity(np_small)
check2 = cosine_similarity(df_np_small)
print(check)
print('\n')
print(check2)

#%% check
print(df_np_small)
#%% print numpy array
for row in np_small:
    rating = row[0]
    movie = row[1]
    print(rating)
    print(movie)

#%%
for pair in combinations(np_small[:,],2):
    x1 = pair[0]
    x2 = pair[1]
    print(cos_sim(x1,x2))

# group_by movie id -> then put ratings for group into a list
# run cosine similarity on combinations based on movie_id
#%%
df.shape

#%% check 3
check3 = cosine_similarity(np_mov_rat)
#%% the big guy
the_win = np.array()

for pair in combinations(np_mov_rat[:,],2):
    x1 = pair[0]
    x2 = pair[1]
    np.append.(cos_sim(x1,x2))


#%%


#%%
