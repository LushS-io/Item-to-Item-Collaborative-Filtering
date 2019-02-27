# Homework 2 programming portion
#%% imports
import pandas as pd
import numpy as np
import math as mth
from itertools import combinations
from dfply import *

#%% import datasets
links = pd.read_csv("./movie-lens-data/links.csv")
movies = pd.read_csv("./movie-lens-data/movies.csv")
ratings = pd.read_csv("./movie-lens-data/ratings.csv")
tags = pd.read_csv("./movie-lens-data/tags.csv")

#%% list attributes
print(list(links))
print(list(movies.columns.values)) #another way to list
print(list(ratings))
print(list(tags))

#%% join movies and ratings
# movies.join(other=ratings,on='movieId',lsuffix="_movies",rsuffix="_ratings")
df = pd.merge(movies,ratings, on='movieId', how='outer')

#%%
