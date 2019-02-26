#%% Import
import pandas as pd
import numpy as np
from math import exp, sqrt, pow
from sklearn.metrics.pairwise import cosine_similarity

#%% define dataset
df = pd.read_csv("/Users/troy/WSU_SPRING_2019/DATA_MINING/hw2/Item to Item Collaborative Filtering/similaritymatrixdebugginginformation.csv")
df

#%%
df_test = df.drop(['Unnamed: 0'],axis=1)
#cosine_similarity(df_test)
df_test
df_test_results = cosine_similarity(df_test)
print(df_test_results)
type(df_test_results)

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