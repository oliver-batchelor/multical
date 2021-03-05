from os import path
import os
from natsort import natsorted


image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)



def find_image_files(filepath, extensions=image_extensions):
    return [filename for filename in natsorted(os.listdir(filepath))
                if has_extension(extensions, path.join(filepath, filename))]

def find_unmatched_files(camera_paths, extensions):
  return {k:find_image_files(d, extensions) for k, d in camera_paths.items()}

def find_matching_files(camera_paths, extensions):
  camera_files = find_unmatched_files(camera_paths, extensions)
  return natsorted(set.intersection(*[set(files) for files in camera_files.values()]))


def find_cameras(base_dir, cameras, camera_pattern, extensions=image_extensions):
  if cameras is None:
    cameras = natsorted(find_nonempty_dirs(base_dir, extensions))

  camera_pattern = camera_pattern or "{camera}" 
  return {camera:path.join(base_dir, camera_pattern.format(camera=camera)) for camera in cameras}

  


def find_nonempty_dirs(filepath, extensions=image_extensions):
    return [local_dir for local_dir in os.listdir(filepath)
      for abs_dir in [path.join(filepath, local_dir)]
      if path.isdir(abs_dir) and len(find_image_files(abs_dir, extensions)) > 0 
    ]



def find_images(camera_dirs, extensions=image_extensions):
  image_names = find_matching_files(camera_dirs, extensions)
  return image_names, filenames(camera_dirs.values(), image_names)

def filenames(camera_dirs, image_names):
  return [[path.join(camera_dir, image) for image in image_names]
    for camera_dir in camera_dirs]
