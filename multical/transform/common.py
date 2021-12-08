from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
import numpy as np

from collections import Counter

def cluster_common(vectors, min_clusters=3, cluster_size=10):
    Z = linkage(whiten(vectors), 'ward')
    n_clust = max(vectors.shape[0] / cluster_size, min_clusters)

    clusters = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusters[clusters >= 0])
    most = cc.most_common(n=1)[0][0]

    return clusters == most
 

def mean_robust(vectors):
  common_indexes = cluster_common(vectors)
  return vectors[common_indexes].mean(axis=0) if len(vectors) > 1\
    else vectors[0]



def sample_furthest_distances(distances, n):
    num_points = distances.shape[0]
    
    first = np.random.randint(num_points)
    indexes = [first]
    ds = distances[first, :]

    for _ in range(1, n):
        index = np.argmax(ds)
        indexes.append(index)
        ds = np.minimum(ds, distances[index, :])

    return np.array(indexes)

