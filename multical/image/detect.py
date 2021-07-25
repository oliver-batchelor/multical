from functools import partial
import os.path as path
import cv2

from multiprocessing import Pool                                          

import numpy as np
from multical.camera import  stereo_calibrate
from multical.threading import parmap_lists

from structs.struct import transpose_structs, struct, filter_none

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


def load_images(filenames, prefix=None, **map_options):
    if prefix is not None:
      filenames = [[path.join(prefix, file) for file in camera_files]
        for camera_files in filenames]

    return parmap_lists(load_image, filenames, **map_options)

def detect_image(image, boards):
  return [board.detect(image) for board in boards]

def detect_images(boards, images, **map_options):
  detect = partial(detect_image, boards=boards)
  return parmap_lists(detect, images, **map_options, pool=Pool)


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
  


      
