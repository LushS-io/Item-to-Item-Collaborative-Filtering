import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

A = np.array(
[[4,5,0,5,1,0],
[0,3,4,3,1,2],
[2,0,1,3,0,4]])

dist_out = 1-pairwise_distances(A, metric="cosine")
dist_out


