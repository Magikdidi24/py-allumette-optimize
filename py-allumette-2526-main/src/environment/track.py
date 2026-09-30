# environment/track.py [FULLY PROVIDED TO STUDENTS]
import numpy as np
from scipy.interpolate import splprep, splev

class Track:
    def __init__(self):
        waypoints = np.array([
            [0.0,  0.0],
            [4.0,  0.0],
            [6.0, -1.0],
            [7.0, -3.0],
            [5.0, -5.0],
            [3.0, -4.0],
            [2.0, -2.0],
            [-1.0, -1.5],
            [-4.0, -3.0],
            [-6.0,  0.0],
            [-3.0,  2.0],
            [-1.0,  0.5],
            [0.0,  0.0]
        ])

        x_pts, y_pts = waypoints[:, 0], waypoints[:, 1]
        tck, _ = splprep([x_pts, y_pts], s=0, per=True)
        u_new = np.linspace(0, 1, 500)
        self.x_center, self.y_center = splev(u_new, tck)
        self.width = 1.0

        dx = np.gradient(self.x_center)
        dy = np.gradient(self.y_center)
        norms = np.hypot(dx, dy)
        nx = -dy / norms
        ny =  dx / norms

        self.x_inner = self.x_center - self.width / 2 * nx
        self.y_inner = self.y_center - self.width / 2 * ny
        self.x_outer = self.x_center + self.width / 2 * nx
        self.y_outer = self.y_center + self.width / 2 * ny

        # Arc-length parameterisation (used by MPC cost)
        diffs = np.diff(np.column_stack([self.x_center, self.y_center]), axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
        self.total_length = self.arc_lengths[-1]

    def get_closest_idx(self, state):
        """Returns the index on the centerline closest to the car."""
        car_x, car_y = state[0], state[1]
        distances = np.hypot(self.x_center - car_x, self.y_center - car_y)
        return int(np.argmin(distances))

    def get_reference(self, state, lookahead_steps=20):
        """Returns the next N centerline waypoints ahead of the car."""
        idx = self.get_closest_idx(state)
        indices = [(idx + i) % len(self.x_center) for i in range(lookahead_steps)]
        ref_x = self.x_center[indices]
        ref_y = self.y_center[indices]
        return ref_x, ref_y

    def is_on_track(self, state):
        """Returns False if the car is outside the track boundaries."""
        idx = self.get_closest_idx(state)
        car_x, car_y = state[0], state[1]
        cx = self.x_center[idx]
        cy = self.y_center[idx]
        dist_to_center = np.hypot(car_x - cx, car_y - cy)
        return dist_to_center < self.width / 2