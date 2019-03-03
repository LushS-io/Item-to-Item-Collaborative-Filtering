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
from scipy.sparse import csr_matrix
#%% Methods
"""Takes 2 vectors a, b and returns the cosine similarity according to the definition of the dot product"""
def cos_sim(a, b):
    dot_product = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def cos_sim_nd(nd_array,axis=0):
    nd_array = np.array(nd_array)     
    if axis < 0:
        axis += a.ndim
    indexs = np.arange(nd_array.shape[axis])
    data_type = np.dtype()
    dot_product = np.apply_along_axis()
    dot_product = np.dot(nd_array,)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def combs_nd(a, r, axis=0):
    a = np.asarray(a)
    if axis < 0:
        axis += a.ndim
    indices = np.arange(a.shape[axis])
    dt = np.dtype([('', np.intp)]*r)
    indices = np.fromiter(combinations(indices, r), dt)
    indices = indices.view(np.intp).reshape(-1, r)
    return np.take(a, indices, axis=axis)

def import_hw2data_to_pd_df ():
    # dataset from hw 2
    user_1 = np.array([4,5,0,5,1,0]) 
    user_2 = np.array([0,3,4,3,1,2])
    user_3 = np.array([2,0,1,3,0,4])

    # print combination of similarities
    print(cos_sim(user_1, user_2))
    print(cos_sim(user_1,user_3))
    print(cos_sim(user_2,user_3))
    #%% Create pd_df from array data
    '''Append 3 users into df'''
    users = pd.DataFrame(data=(user_1,user_2,user_3))
    return users

#%% Create pandas and numpy df
users = import_hw2data_to_pd_df()
df_users = users
np_users = users.to_numpy()

#%% playground

x = np.array(range(18))
x = np.reshape(x,(3,6))
print(x)

np.take(x, 0, 0)
np.take(x, 1, 0)

print(x.dtype)

np_dot_prod = np.dot(x[0],x[1])
a_norm = np.linalg.norm(x[0])

print(a_norm)
print(np_dot_prod)





#%% playground 2
print(cosine_similarity(np_users))
NP_COMBO_TEST = combs_nd(np_users,2)
# x[0,1]
print('\n{}'.format(cos_sim(x[0,0],x[0,1])))
np.take(x, 2, 0)
# test_dot_product = np.dot(np.take(x,0,0),np.take(x,1,0))

#%% sparse matrix from np
np_sprase = scipy.sparse.csr_matrix(np_users)
print("Check Validity\n\n{}\n\nThe Sparse Matrix\n{}".format(
    np_sprase.check_format, np_sprase))

# try on sparse matrix
# cosine_similarity(np_sprase)
cos_sim(x[0,0],x[0,1]) # okay it works on np array, now how to scale

#%% play with np iterators
y = np.asarray(x)
row = np.arange(x.shape[0])
col = np.arange(x.shape[1])
print('index = {}'.format(col))
# inx = np.fromiter(x,int)

#%% sprase matrix from pd
pd_sparse = scipy.sparse.csr_matrix(df_users.values)
print("Check Validity\n\n{}\n\nThe Sparse Matrix\n{}".format(pd_sparse.check_format,pd_sparse))

#%% get combos
combos = combinations(np_users[:,],2) #get combos of np_array
print("Getting combos...and type = {}".format(type(combos)))
print("Look inside combos => \n")
print(list(combos))
print("Convert list into np.array")
np_combos = np.asarray(list(combos))
print("Here's the np_combos array...\n{}".format(np_combos))

#%%
type(np.array(list(combos))) # check combo to nparray convert
np_combos = np.array(list(combos)) # set np combo array 
sparseNPcombos = scipy.sparse.csr_matrix(np_combos)# to sprase_matrix
# print(type(sparseNPcombos)) #check type
# print(sparseNPcombos[:,:])#check 
print(list(combos))
#%% get combo for sparse
# combo_sprase = combinations(sprase_matrix([:,[0]]))
combo_sprase = combinations(sprase_matrix[:,[0]],2)
print( np.array( list( combinations( combo_sprase[:,],2))))
#%% play sparse matrix
print(sprase_matrix[:,0] )
#%% run cos_sim on csr_sparse()
cos_sim(sprase_matrix())
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
