import numpy as np


class Point2d():


    def __init__(self, x, y, r, g, b, z=None, idx_pt=None):
        self._pt = (x, y)
        self._rgb = (r,g,b)
        self._idx_pt = idx_pt
        self._depth = z


    @property
    def depth(self):
        return self._depth


    @property
    def idx_pt(self):
        return self._idx_pt


    @property
    def coor(self):
        return self._pt


    @property
    def color(self):
        return self._rgb


    def __repr__(self):
        return f'<Point ({self._pt[0]}, {self._pt[1]}) <-> ({self._rgb[0]}, {self._rgb[1]}, {self._rgb[2]})>'


    @staticmethod
    def get_list_colors(list_Points):
        list_colors = []
        for point in list_Points:
            list_colors += [color for color in point.color]
        return list_colors


    @staticmethod
    def get_list_coors(list_Points):
        list_coors = []
        for point in list_Points:
            list_coors += [coor for coor in point.coor]
        return list_coors


    def get_closest_point_to_camera(self, list_Points):
        pass


    # check correction positions by depth (z_coors) MAX
    @staticmethod
    def get_color(list_Points, z_coors=None, strategy='mean'):
        if list_Points:
            colors = np.asarray([list(point.color) for point in list_Points])
            if isinstance(z_coors, np.ndarray):
                depths = np.asarray([point._depth for point in list_Points])
                # select only closest point to camera
                depth_min_idx = np.argmin(depths)
                depth_max_idx = np.argmax(depths)
                return colors[depth_max_idx]
            if strategy == 'mean':
                return colors.mean(axis=0)
            elif strategy == 'max':
                return colors.max(axis=0)
            elif strategy == 'min':
                return colors.min(axis=0)
        else:
            # white color
            return (1., 1., 1.)