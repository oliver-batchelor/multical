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


def load_detect(board, filename):
  assert path.isfile(filename), f"load_image: file {filename} does not exist"

  image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  assert image is not None, f"load_image: could not read {filename}"
  
  return struct(images=image, points=board.detect(image))


def common_image_size(images):
  image_shape = images[0].shape
  h, w, *_ = image_shape

  assert all([image.shape == image_shape for image in images])
  return (w, h)

def detect_images(board, filenames, prefix=None, j=len(os.sched_getaffinity(0)) // 2, chunksize=1):
  pool = ThreadPool(processes=j)                                                        

  cam_lengths = map_list(len, filenames)
  flat_files = concat_lists(filenames)

  if prefix is not None:
    flat_files = [path.join(prefix, file) for file in flat_files]
  
  loader = pool.imap(partial(load_detect, board), flat_files, chunksize=chunksize)
  results = list(tqdm(loader, total=len(flat_files)))

  results = transpose_structs(results)._map(split_list, cam_lengths)
  
  return results._extend(image_size = map_list(common_image_size, results.images))




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
  

def refine_detections(board, cameras, points, images):
  return [[board.refine_points(camera, frame_dets, image) 
    for frame_dets, image in zip(cam_dets, cam_images)]
      for camera, cam_dets, cam_images in zip(cameras, points, images)]
      
