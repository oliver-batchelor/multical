import numpy as np

def max_2d(m):
  i = np.unravel_index(np.argmax(m), m.shape)
  return i, m[i]

def select_pairs(overlaps, hop_penalty=0.8):
  """ Greedy spanning tree algorithm, with penalty for depth.
  Prefer shorter paths with hop penalty < 1.

  returns: root, a list of pairs
  """

  n = overlaps.shape[0]

  master = np.argmax(overlaps.sum(1))
  weight = (np.arange(n) == master).astype(np.float32).reshape(n, 1)
  overlaps[:, master] = 0
  pairs = []

  while (len(pairs) + 1 < n):
    (parent, child), overlap  = max_2d(overlaps * weight)
    if overlap <= 0: break

    overlaps[:, child] = 0
    weight[child] = weight[parent] * hop_penalty

    pairs.append( (parent, child) )

  return master, pairs
