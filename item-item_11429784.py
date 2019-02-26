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