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

