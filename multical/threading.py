
from functools import partial
from multiprocessing import Pool, cpu_count, get_logger
from multiprocessing.pool import ThreadPool

from structs.struct import map_list, concat_lists, split_list

from tqdm import tqdm
import traceback

# Shortcut to multiprocessing's logger
def error(msg, *args):
    return get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


def parmap_list(f, xs, j=cpu_count() // 2, chunksize=1, pool=Pool, progress=tqdm):

  with pool(processes=j) as pool:
    iter = pool.imap(LogExceptions(f), xs, chunksize=chunksize)
    
    if progress is not None:
      iter = progress(iter, total=len(xs))

    return list(iter)




def parmap_lists(f, xs_list, j=cpu_count() // 2, chunksize=1, pool=ThreadPool):
  """ Map over a list of lists in parallel by flattening then splitting at the end"""
  cam_lengths = map_list(len, xs_list)
  xs = concat_lists(xs_list)

  results = parmap_list(f, xs, j=j, chunksize=chunksize, pool=pool)
  return split_list(results, cam_lengths)
