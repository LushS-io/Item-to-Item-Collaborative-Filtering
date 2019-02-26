#%% Import
import pandas as pd
import numpy as np
from math import exp, sqrt, pow
from sklearn.metrics.pairwise import cosine_similarity

#%% define dataset
df = pd.read_csv("/Users/troy/WSU_SPRING_2019/DATA_MINING/hw2/Item to Item Collaborative Filtering/similaritymatrixdebugginginformation.csv")
df
