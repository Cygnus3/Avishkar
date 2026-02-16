import math
from utils.geometry import normalize_angle


class Odometry:
    def __init__(self):
        # Ground truth pose
        self.gt_x = 5.0
        self.gt_y = 2.5
        self.gt_theta = 0.0
        

        # Estimated pose (odometry)
        self.x = 5.0
        self.y = 2.5
        self.theta = 0.0 

    def update(self, v, omega, dt):
        """
        Update ground truth and odometry states
        using commanded linear and angular velocity.
        """

        # --------------------------------
        # Ground truth motion integration
        # --------------------------------
        self.gt_x += v * math.cos(self.gt_theta) * dt
        self.gt_y += v * math.sin(self.gt_theta) * dt
        self.gt_theta += omega * dt
        self.gt_theta = normalize_angle(self.gt_theta)

        # --------------------------------
        # Odometry motion integration
        # --------------------------------
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt 
        self.theta = normalize_angle(self.theta)



