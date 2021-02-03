from functools import partial
import os
import os.path as path
import cv2
from natsort.natsort import natsorted
from structs.numpy import shape

from tqdm import tqdm
from multiprocessing.pool import ThreadPool                                                

import numpy as np
from multical.camera import  stereo_calibrate

from structs.struct import concat_lists, transpose_structs, struct, filter_none, split_list, map_list


def load_image(filename):
  assert path.isfile(filename), f"load_image: file {filename} does not exist"

  image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  assert image is not None, f"load_image: could not read {filename}"
  return image

  

def common_image_size(images):
  image_shape = images[0].shape
  h, w, *_ = image_shape

  assert all([image.shape == image_shape for image in images])
  return (w, h)



def par_map_lists(f, xs_list, j=len(os.sched_getaffinity(0)) // 2, chunksize=1):
  """ Map over a list of lists in parallel by flattening then splitting at the end"""
  cam_lengths = map_list(len, xs_list)
  flat_files = concat_lists(xs_list)

  with ThreadPool(processes=j) as pool:
    iter = pool.imap(f, flat_files, chunksize=chunksize)
    results = list(tqdm(iter, total=len(flat_files)))
    return split_list(results, cam_lengths)


def load_images(filenames, prefix=None, **map_options):
    if prefix is not None:
      filenames = [[path.join(prefix, file) for file in camera_files]
        for camera_files in filenames]

    return par_map_lists(load_image, filenames, **map_options)


def detect_images(boards, images, **map_options):
  def detect_image(image):
    return [board.detect(image) for board in boards]

  return par_map_lists(detect_image, images, **map_options)



def intersect_detections(board, d1, d2):
  ids, inds1, inds2 = np.intersect1d(d1.ids, d2.ids, return_indices=True)

  if len(ids) > 0:
    return struct(points1 = d1.corners[inds1], points2 = d2.corners[inds2], 
      object_points = board.points[ids], ids=ids)
  else:
    return None

def stereo_calibrate_detections(detections, board, cameras, i, j, **kwargs):
  matching = [intersect_detections(board, d1, d2) 
    for d1, d2 in zip(detections[i], detections[j])]

  matching_frames = transpose_structs(filter_none(matching))
  return stereo_calibrate((cameras[i], cameras[j]), matching_frames, **kwargs)  
  


      
