from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten

from collections import Counter

def cluster(vectors, min_clusters=3, cluster_size=10):
    Z = linkage(whiten(vectors), 'ward')
    n_clust = max(vectors.shape[0] / cluster_size, min_clusters)

    clusters = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusters[clusters >= 0])
    most = cc.most_common(n=1)[0][0]

    return clusters == most
 

def mean_robust(vectors):
  return vectors[cluster(vectors)].mean(axis=0) if len(vectors) > 1\
    else vectors[0]