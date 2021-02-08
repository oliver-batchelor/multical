from multical.optimization.parameters import Parameters
from multical import tables


class Static(Parameters):
  def __init__(self, poses):
    self.poses = poses


  def projection(self, cameras, camera_poses, points, detections=None):


    tables.expand_views(camera_poses, self.poses)


    image_points = [camera.project(p) for camera, p in 
      zip(self.cameras, transformed.points)]

    return Table.create(points=np.stack(image_points), valid=transformed.valid)

  

    
  def params(self):
    pass

  def with_params(self, params):
    pass


  def sparsity(self, index_table):
    pass