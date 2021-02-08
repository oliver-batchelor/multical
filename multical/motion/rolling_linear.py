  # @cached_property
  # def times(self):
  #   image_heights = np.array([camera.image_size[1] for camera in self.cameras])    
  #   return self.point_table.points[..., 1] / np.expand_dims(image_heights, (1,2,3))

  # @cached_property
  # def transformed_rolling(self):
  #   poses = self.pose_estimates
  #   start_frame = np.expand_dims(poses.rig.poses, (0, 2, 3))
  #   end_frame = np.expand_dims(self.frame_motion.poses, (0, 2, 3))
    
  #   frame_poses = interpolate_poses(start_frame, end_frame, self.times)
  #   view_poses = np.expand_dims(poses.camera.poses, (1, 2, 3)) @ frame_poses  

  #   board_points = self.stacked_boards
  #   board_points_t = matrix.transform_homog(t = np.expand_dims(poses.board.poses, 1), points = board_points.points)

  #   return struct(
  #     points = matrix.transform_homog(t = view_poses, points = np.expand_dims(board_points_t, (0, 1))),
  #     valid = self.valid
  #   )

 
  #  @cached_property
  # def transformed_rolling_linear(self):
  #   poses_start = self.pose_estimates
  #   poses_end = poses_start._extend(rig=self.frame_motion)

  #   table_start = tables.expand_poses(poses_start)
  #   table_end = tables.expand_poses(poses_end)

  #   start_frame = transform_points(table_start, self.stacked_boards)
  #   end_frame = transform_points(table_end, self.stacked_boards)

  #   return struct(
  #     points = lerp(start_frame.points, end_frame.points, self.times),
  #     valid = start_frame.valid & end_frame.valid
  #   )