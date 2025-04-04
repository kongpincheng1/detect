left_world = self.pixel_to_world(x1, cy_pixel, depth_at_center)
right_world = self.pixel_to_world(x2, cy_pixel, depth_at_center)
real_diameter = math.sqrt((left_world[0] - right_world[0])**2 + (left_world[1] - right_world[1])**2)
