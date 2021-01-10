import os
import os.path as path
import cv2
from natsort.natsort import natsorted
from structs.numpy import shape

from tqdm import tqdm
from multiprocessing import Pool                                                

import numpy as np
from multical.camera import  stereo_calibrate

from structs.struct import transpose_structs, struct, filter_none


def load_image(filename):
  assert path.isfile(filename), f"load_image: file {filename} does not exist"

  image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  assert image is not None, f"load_image: could not read {filename}"
  
  return image


def detect_images(board, camera_names, filenames, show=False, j=4, prefix=None):
  pool = Pool(processes=j)                                                        
 
  def detect_camera(k, image_files):
    print(f"Detecting boards in camera {k}")
    if prefix is not None:
      image_files = [path.join(prefix, file) for file in image_files]

    image_shape = None
    detections = []
    images = []

    loader = pool.imap(load_image, image_files, chunksize=j)
    for image in tqdm(loader, total=len(image_files)):
      if image_shape is None:
        image_shape = image.shape
      detections.append(board.detect(image))
      images.append(image) 

    height, width = image_shape
    return struct(points=detections, image_size=(width, height), images=images)

  detections = [detect_camera(k, image_files) 
    for k, image_files in zip(camera_names, filenames)]

  return transpose_structs(detections)


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
      
