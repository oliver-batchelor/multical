
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from structs.struct import map_list, concat_lists, split_list

from tqdm import tqdm

def map_lists(f, xs_list, j=cpu_count() // 2, chunksize=1, pool=ThreadPool):
  """ Map over a list of lists in parallel by flattening then splitting at the end"""
  cam_lengths = map_list(len, xs_list)
  flat_files = concat_lists(xs_list)

  with pool(processes=j) as pool:
    iter = pool.imap(f, flat_files, chunksize=chunksize)
    results = list(tqdm(iter, total=len(flat_files)))
    return split_list(results, cam_lengths)