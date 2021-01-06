import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten

from collections import Counter
from scipy.spatial.transform import Rotation as R

from . import matrix


def transform_points(points, rtvec):
  rvec, tvec = split(rtvec)
  return R.from_rotvec(rvec).apply(points) + tvec


def split(rtvec):
  assert rtvec.shape[-1] == 6
  return rtvec[..., 0:3], rtvec[..., 3:6]

def join(rvec, tvec):
  assert rvec.shape[-1] == 3 and tvec.shape[-1] == 3
  return np.hstack([rvec, tvec])

def to_matrix(rtvec):
  rvec, tvec = split(rtvec)
  rotation = R.from_rotvec(rvec).as_matrix()
  return matrix.join(rotation, tvec)

def from_matrix(m):
  rot, t = matrix.split(m)
  rvec = R.from_matrix(rot).as_rotvec()
  return join(rvec, t)


def multiply(rtvecs1, rtvecs2):
  m = to_matrix(rtvecs1) @ to_matrix(rtvecs2)
  return from_matrix(m)


def relative_to(source, dest):
  m = matrix.relative_to(to_matrix(source), to_matrix(dest))
  return from_matrix(m)


def as_rtvec(extrinsic):
  if extrinsic is None:
    return np.zeros(6)
  elif extrinsic.shape == (4, 4):
    return from_matrix(extrinsic)          
  elif extrinsic.shape == (6,):
    return extrinsic
  else:
    assert False, f"expected extrinsic of shape 6 or 4x4, got: {extrinsic.shape}"


def cluster(vectors, min_clusters=3, cluster_size=10):
    Z = linkage(whiten(vectors), 'ward')
    n_clust = max(vectors.shape[0] / cluster_size, min_clusters)

    clusters = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusters[clusters >= 0])
    most = cc.most_common(n=1)[0][0]

    return clusters == most
 

def mean_robust(rtvecs):
  return rtvecs[cluster(rtvecs)].mean(axis=0) if len(rtvecs) > 1\
    else rtvecs[0]
