from os import path
import os
from natsort import natsorted


image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)



def find_image_files(filepath, extensions=image_extensions):
    return [filename for filename in natsorted(os.listdir(filepath))
                if has_extension(extensions, path.join(filepath, filename))]

def find_matching_files(filepath, cameras, extensions):
  abs_dirs = [path.join(filepath, camera_name) for camera_name in cameras]
  camera_files = [find_image_files(d, extensions) for d in abs_dirs]

  return natsorted(set.intersection(*[set(files) for files in camera_files]))
    

def find_nonempty_dirs(filepath, extensions=image_extensions):
    return [local_dir for local_dir in os.listdir(filepath)
      for abs_dir in [path.join(filepath, local_dir)]
      if path.isdir(abs_dir) and len(find_image_files(abs_dir, extensions)) > 0 
    ]

def find_images(base_dir, cameras=None, extensions=image_extensions):
  if cameras is None:
    cameras = find_nonempty_dirs(base_dir, extensions)

  camera_names = natsorted(cameras)
  image_names = find_matching_files(base_dir, cameras, extensions)

  return camera_names, image_names, filenames(base_dir, camera_names, image_names)

def filenames(base_dir, camera_names, image_names):
  return [[path.join(base_dir, camera, image) for image in image_names]
    for camera in camera_names]
