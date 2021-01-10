from multiprocessing import Pool
import cv2
from tqdm import tqdm


def split_list(xs, sizes):
  assert len(xs) == sum(sizes)
  splits = []
  for size in sizes:
    splits.append(xs[:size])
    xs = xs[size:] 

  return splits

def undistort_image(args):
  image, undistort_map = args
  return cv2.remap(image, undistort_map, None, cv2.INTER_CUBIC)


def undistort_images(images, cameras, j=4):
  pool = Pool(processes=j)                                                        
 
  image_pairs = [(image, camera.undistort_map) 
    for camera, cam_images in zip(cameras, images)
      for image in cam_images]
  
  loader = pool.imap(undistort_image, image_pairs, chunksize=j)
  undistorted = list(tqdm(loader, total=len(image_pairs)))

  return split_list(undistorted, [len(i) for i in images])
