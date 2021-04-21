from apriltags_eth import make_default_detector
from collections import namedtuple

DetectionResult = namedtuple(
    'DetectionResult', ['success', 'image_points', 'target_points', 'ids'])


class AprilGridDetector(object):
  """
  https://github.com/safijari/apriltags2_ethz/blob/master/aprilgrid/__init__.py
  Modification to support AprilGrid start_id so that we can use multiple
  board

  Just like the source, this only works with tagCodes36h11 because the python
  binding of make_default_detector() doesn't accept any parameter
  """
  def __init__(self, rows, columns, size, spacing, start_id=0):
    assert size != 0.0
    assert spacing != 0.0
    self.rows = rows
    self.columns = columns
    self.size = size
    self.spacing = spacing
    self.start_id = start_id
    self.detector = make_default_detector()

  def is_detection_valid(self, detection, image):
    d = detection
    h, w = image.shape[0:2]
    for cx, cy in d.corners:
      if cx < 0 or cx > w:
        return False
      if cy < 0 or cy > h:
        return False
    if not d.good:
      return False
    if d.id < self.start_id:
      return False
    if d.id >= self.rows * self.columns + self.start_id:
      return False

    return True

  def get_tag_corners_for_id(self, tag_id):
    # order is lower left, lower right, upper right, upper left
    # Note: tag_id of lower left tag is 0, not 1
    a = self.size
    b = self.spacing * a
    tag_row = (tag_id) // self.columns
    tag_col = (tag_id) % self.columns
    left = bottom = lambda i: i * (a + b)
    right = top = lambda i: (i + 1) * a + (i) * b
    return [(left(tag_col), bottom(tag_row)),
            (right(tag_col), bottom(tag_row)),
            (right(tag_col), top(tag_row)), (left(tag_col), top(tag_row))]

  def compute_observation(self, image):
    # return imagepoints and the coordinates of the corners
    # 1. remove non good tags
    detections = self.detector.extract_tags(image)

    # Duplicate ID search
    ids = {}
    for d in detections:
      if d.id in ids:
        raise Exception("There may be two physical instances of the same tag in the image")
      ids[d] = True

    filtered = [d for d in detections if self.is_detection_valid(d, image)]

    image_points = []
    target_points = []
    ids = []

    filtered.sort(key=lambda x: x.id)

    # TODO: subpix refinement?
    for f in filtered:
      # new id starting from 0 to not break anything else in the codebase
      id = f.id - self.start_id
      target_points.extend(self.get_tag_corners_for_id(id))
      image_points.extend(f.corners)
      ids.extend([id, id, id, id])

    success = True if len(filtered) > 0 else False

    return DetectionResult(success, image_points, target_points, ids)
