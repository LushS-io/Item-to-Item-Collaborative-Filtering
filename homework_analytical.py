# Homework 2 analytical for cs315 
#%% Import
import pandas as pd
import numpy as np
import math as mth
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from dfply import *
from itertools import starmap
import scipy as scipy
import time as time 

#%% define dataset
df = pd.read_csv("/Users/troy/WSU_SPRING_2019/DATA_MINING/hw2/Item to Item Collaborative Filtering/similaritymatrixdebugginginformation.csv")
df = df.drop(['Unnamed: 0'],axis=1) #remove first column
# side note: axis=0 => row
#            axis=1 => column 

#%%
'''Test personal results against sklearn'''
cos_sim_correct_results = cosine_similarity(df)
print(cos_sim_correct_results)

#%%
"""Takes 2 vectors a, b and returns the cosine similarity according to the definition of the dot product"""
def cos_sim(a, b):
    dot_product = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# dataset from hw 2
user_1 = np.array([4,5,0,5,1,0]) 
user_2 = np.array([0,3,4,3,1,2])
user_3 = np.array([2,0,1,3,0,4])

# print combination of similarities
print(cos_sim(user_1, user_2))
print(cos_sim(user_1,user_3))
print(cos_sim(user_2,user_3))
#%%
'''Append 3 users into df'''
users = pd.DataFrame(data=(user_1,user_2,user_3))
df_users = users
users = users.to_numpy()
#%% sprase matrix
sprase_matrix = scipy.sparse.csr_matrix(df_users.values)
print(sprase_matrix)
#%%
print(users[:,[0]])
#%%
for pair in combinations(users[:,],2):
    x1 = pair[0]
    x2 = pair[1]
    print(cos_sim(x1,x2))

#%% test 
sample_cos_sim_correct_results = cosine_similarity(users)
sample_cos_sim_correct_results

#%% do with lamba apply
combos = combinations(users[:,],2) 
type(combos)
# test['cos_sim'] = combinations(users[:,],2).apply(lambda row: cos_sim(row[0],row[1]))
#%% convert df into numpy array
numpy_matrix = df.to_numpy()
# pd.__version__
# to_numpy is better to use than df.values as it will soon be depreciated


#%%
# wrangle the data using pandas
## note: np.nan = NaN "not a number" which is basically null
'''Calculate cos_sim of first two rows on actual dataset'''
for pair in combinations(numpy_matrix[:,],2):
    x1 = pair[0]
    x2 = pair[1]
    print(cos_sim(x1,x2))

#%%
def my_func(p):
    return np.sum(p)
my_array = np.array(range(100))
my_combin = np.array(tuple(combinations(my_array, 2)))
my_result = [my_func(pair) for pair in my_combin]
print(my_result)
#%%
users
#%%
df_users
#%% adds up the entire or col of numpy array
np_arr_play = np.apply_along_axis(func1d=sum,axis=1,arr=users)
print(np_arr_play)

#%% pandas df add col 1 and col 2 for each row
df_play = df_users.apply(lambda x: cos_sim(x[0],x[1]),axis=1) 
print(df_play)


#%%
# results = list(itertools.starmap(pow, [(2,5), (3,2), (10,3)]))
results = starmap(pow, [(2,5), (3,2), (10,3)])
#%%
results

#%%
