import numpy as np

class WaypointWorld():
    def __init__(self):
        # (x, y)
        self.state = np.array([0, 0])
        self.xmin = 0
        self.xmax = 10
        self.ymin = 0
        self.ymax = 10

    def reset(self):
        self.state = np.array([0, 0])

    def observe(self):
        return self.state

    def perform(self, action):
        """
        [0, 1, 2, 3] => down, up, right, left
        """
        x, y = self.state
        if action == 0: # down
            self.state = np.array([x, max(self.ymin, y - 1)])
        elif action == 1: # up
            self.state = np.array([x, min(self.ymax, y + 1)])
        elif action == 2: # right
            self.state = np.array([min(self.xmax, x + 1), y])
        elif action == 3: # left
            self.state = np.array([max(self.xmin, x - 1), y])
        else:
            raise ValueError(f'Unexpected action {action}')

        if (self.state == np.array([10, 10])).all():
            r = 100
            self.reset()
        else:
            r = -1

        return r, self.state
