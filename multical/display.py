import cv2
import numpy as np

from .transform import rtvec
from . import tables

from structs.struct import struct
from .image.display import display_stacked
import palettable.colorbrewer.qualitative as palettes


color_sets = struct(
  set1 = np.array(palettes.Set1_9.colors) / 255,
  paired = np.array(palettes.Paired_12.colors) / 255
)


def draw_board_detections(image, detections, color, thickness=1, radius=10, show_ids=True):
  r, g, b = color
  color = np.array([b, g, r]) * 255
  
  for id, corner in zip(detections.ids, detections.corners):
    cv2.drawMarker(image, tuple([int(corner[0]), int(corner[1])]), color=color, markerSize=radius*2, thickness=int(thickness), line_type=cv2.LINE_AA)
    x, y = corner
    if show_ids:
      cv2.putText(image, str(id), (int(x + 2), int(y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 
        radius/40, color=color, lineType=cv2.LINE_AA)
  return image


def show_detections(image, detections, **options):
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  return draw_detections(image, detections, **options)


def draw_detections(image, detections, **options):
  for color, board_detections in zip(color_sets['set1'], detections):
    draw_board_detections(image, board_detections, color, **options)
  return image


# def annotate_projection(board, image, camera, image_table, radius=20.0, thickness=1, scale=1):
  
#   projected = camera.project(board.adjusted_points, image_table.poses).astype(np.float32)

#   image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#   image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))

#   for proj, corner, valid, inlier in zip(projected * scale, 
#       image_table.points * scale, image_table.valid, image_table.inliers):

#     color = (255, 255, 0) if not valid\
#       else (0, 255, 0) if inlier else (0, 0, 255) 

#     cv2.drawMarker(image, tuple(proj), color=color, thickness=int(1), markerSize=int(radius * scale), line_type=cv2.LINE_AA)
#     if valid:
#       cv2.line(image, tuple(proj), tuple(corner), color=(0, 128, 192), thickness=thickness, lineType=cv2.LINE_AA)   


#   rvec, tvec = rtvec.split(rtvec.from_matrix(image_table.poses))
#   cv2.aruco.drawAxis(image, camera.scale_image(scale).intrinsic, 
#     camera.dist, rvec, tvec, board.square_length * 3 * scale)

#   return image


# def display_pose_projections(point_table, pose_table, board, cameras, images, inliers=None, scale=2):
#   inliers = choose(inliers, point_table.valid)
#   table = point_table._merge(pose_table)._extend(inliers = inliers)

#   image_frames = transpose_lists(images)
#   for frame_images, frame_table in zip(image_frames, table._sequence(1)):
#     annotated = [annotate_projection(board, image, camera, image_table, scale=scale) 
#        for camera, image, image_table in zip(cameras, frame_images, frame_table._sequence())]

#     display_stacked(annotated)


# def display_points(point_table, images):

#   image_frames = transpose_lists(images)
#   for frame_images, frame_points in zip(image_frames, point_table._sequence(1)):
#     annotated = [annotate_points(image, points) 
#         for image, points in zip(frame_images, frame_points._sequence())]

#     display_stacked(annotated)


def display_boards(boards, pixels_mm=1):
      
  board_images = [board.draw(pixels_mm=pixels_mm) for board in boards]
  display_stacked(board_images)