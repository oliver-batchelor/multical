import os
import os.path as path
import cv2
from natsort.natsort import natsorted
from structs.numpy import shape

from tqdm import tqdm
from multiprocessing import Pool                                                

import numpy as np
from .camera import  stereo_calibrate

from structs.struct import transpose_structs, struct, filter_none

image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)



def find_image_files(filepath, extensions=image_extensions):
    return [filename for filename in natsorted(os.listdir(filepath))
                if has_extension(extensions, path.join(filepath, filename))]

def find_matching_files(filepath, cameras, extensions):
  abs_dirs = [path.join(filepath, camera_name) for camera_name in cameras]
  camera_files = [find_image_files(d, extensions) for d in abs_dirs]

  matching = sorted(set.intersection(*[set(files) for files in camera_files]))
  return [{camera_name: path.join(camera_name, file)  for camera_name in cameras}
     for file in matching]
    

def find_nonempty_dirs(filepath, extensions=image_extensions):
    return [local_dir for local_dir in os.listdir(filepath)
      for abs_dir in [path.join(filepath, local_dir)]
      if path.isdir(abs_dir) and len(find_image_files(abs_dir, extensions)) > 0 
    ]

def find_images(filepath, cameras=None, extensions=image_extensions):
  if cameras is None:
    cameras = find_nonempty_dirs(filepath, extensions)

  cameras = sorted(cameras)
  images = find_matching_files(filepath, cameras, extensions)
  return cameras, images


def load_image(filename):
  assert path.isfile(filename), f"load_image: file {filename} does not exist"

  image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  assert image is not None, f"load_image: could not read {filename}"
  
  return image


def detect_images(board, filenames, show=False, j=4, prefix=None):
  camera_files = transpose_structs(filenames)
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
    for k, image_files in camera_files.items()]

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
      
