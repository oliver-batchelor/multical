import cv2
import numpy as np

from .transform import rtvec
from . import tables

from structs.struct import transpose_lists, choose


def to_color(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        return image
    elif image.shape[2] == 4:
        return image[:, :, 0:3]
    else:
        assert False, "Unknown image shape: " + str(image.shape)

def stack_images(images, resize_height=None, rotate=0):

    def scale_height(image):
        if rotate > 0:
            image = np.rot90(image, k=rotate)

        if resize_height != image.shape[0]:
            scale = resize_height / image.shape[0]
            image = cv2.resize(image, (int(image.shape[1] * scale), resize_height))

        return to_color(image)

    resize_height = resize_height or images[0].shape[0]

    images = [scale_height(image) for image in images]
    return np.hstack(images)


def display_stacked(images, resize_height=None, rotate=0, resizeable=True):
    image = stack_images(images, resize_height, rotate)
    display(image, resizeable=resizeable)



def display(t, name="image", resizeable=True, height=800):
    cv2.namedWindow(name, flags=cv2.WINDOW_NORMAL if resizeable else cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, t)

    scale = height / t.shape[0]
    cv2.resizeWindow(name, int(t.shape[1] * scale), height)

    while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) > 0:
        keyCode = cv2.waitKey(1)
        if keyCode >= 0:
            return keyCode

    return -1



def annotate_points(image, points):
  detections = tables.sparse_points(points)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  for corner, id in zip(detections.ids, detections.corners):
    cv2.drawMarker(image, tuple(corner), color=(0, 255, 0), thickness=int(1), line_type=cv2.LINE_AA)

    x, y = corner
    cv2.putText(image, str(id), (int(x + 10), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
      0.5, color=(255, 255, 0), lineType=cv2.LINE_AA)

  return image


def annotate_projection(board, image, camera, image_table, radius=20.0, thickness=1, scale=1):
  
  projected = camera.project(board.adjusted_points, image_table.poses).astype(np.float32)

  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))

  for proj, corner, valid, inlier in zip(projected * scale, 
      image_table.points * scale, image_table.valid_points, image_table.inliers):

    color = (255, 255, 0) if not valid\
      else (0, 255, 0) if inlier else (0, 0, 255) 

    cv2.drawMarker(image, tuple(proj), color=color, thickness=int(1), markerSize=int(radius * scale), line_type=cv2.LINE_AA)
    if valid:
      cv2.line(image, tuple(proj), tuple(corner), color=(0, 128, 192), thickness=thickness, lineType=cv2.LINE_AA)   


  rvec, tvec = rtvec.split(rtvec.from_matrix(image_table.poses))
  cv2.aruco.drawAxis(image, camera.scale_image(scale).intrinsic, 
    camera.dist, rvec, tvec, board.square_length * 3 * scale)

  return image

def display_pose_projections(point_table, pose_table, board, cameras, images, inliers=None, scale=2):
  inliers = choose(inliers, point_table.valid_points)
  table = point_table._merge(pose_table)._extend(inliers = inliers)

  image_frames = transpose_lists(images)
  for frame_images, frame_table in zip(image_frames, table._sequence(1)):
    annotated = [annotate_projection(board, image, camera, image_table, scale=scale) 
       for camera, image, image_table in zip(cameras, frame_images, frame_table._sequence())]

    display_stacked(annotated)


def display_points(point_table, images):

  image_frames = transpose_lists(images)
  for frame_images, frame_points in zip(image_frames, point_table._sequence(1)):
    annotated = [annotate_points(image, points) 
        for image, points in zip(frame_images, frame_points._sequence())]

    display_stacked(annotated)


      
