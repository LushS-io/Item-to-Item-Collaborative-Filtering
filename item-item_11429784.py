#%% Import
import pandas as pd
import numpy as np
import math as mth
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


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
users = users.to_numpy()
#%%
print(users[:,[0]])
#%%
for pair in combinations(users[:,],2):
    x1 = pair[0]
    x2 = pair[1]
    print(cos_sim(x1,x2))

#%% convert df into numpy array
numpy_matrix = df.to_numpy()
# pd.__version__
# to_numpy is better to use than df.values as it will soon be depreciated


#%%
# wrangle the data using pandas
## note: np.nan = NaN "not a number" which is basically null
'''Calculate cos_sim of first two rows on actual dataset'''
x1 = numpy_matrix[0]
x2 = numpy_matrix[1] 
cos_sim(x1,x2)