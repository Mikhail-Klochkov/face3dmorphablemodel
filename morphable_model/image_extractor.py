import logging
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class ImageExtractor():

    filled_squared_window = 2

    @staticmethod
    def extract_image_another(image_points, rgb_points, height_num_pixels=256, return_uint_image=True,
                              visible_idx_pts=None, fill_pixels=True):
        if not (isinstance(image_points, np.ndarray) and image_points.shape[1] == 2):
            raise ValueError(f'Shape: {image_points.shape}')
        h = height_num_pixels
        pt_max, pt_min = image_points.max(axis=0), image_points.min(axis=0)
        y_max, x_max = pt_max
        y_min, x_min = pt_min
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        # resize for balance
        w = int(h * (delta_x / delta_y))
        image = np.ones((h, w, 3))
        print(f'Image shape: {(h, w)}')
        x_range = np.linspace(x_min - delta_x * 0.2, x_max + delta_x * 0.2, w + 1)
        y_range = np.linspace(y_min - delta_y * 0.2, y_max + delta_y * 0.2, h + 1)
        # steps of grid
        square_to_points_structure, found_squared_pixs = ImageExtractor._get_square_points_correspondence(image_points,
                                                                                                        rgb_points,
                                                                                                        x_range,
                                                                                                        y_range,
                                                                                                        bin_mask=False)
        points_plot = []
        colors_plot = []
        for key_square, list_points in square_to_points_structure.items():
            # we need ignore not visible points
            # find by hash?
            if visible_idx_pts:
                list_points = ImageExtractor._filtered_non_visible_points(list_points, visible_idx_pts)
            if list_points:
                # pixel position
                idx_y, idx_x = [int(coor) for coor in key_square.split('_')]
                color_square_reqion = Point2d.get_color(list_points, strategy='mean')
                print(color_square_reqion)
                image[idx_y, idx_x] = color_square_reqion
                list_coors = Point2d.get_list_coors(list_points)
                list_colors = Point2d.get_list_coors(list_points)
                colors_plot += list_colors
                points_plot += list_coors
            else:
                # TODO: not implemented
                pass

        # after all
        if fill_pixels:
            image = ImageExtractor._filled_white_pixels(image)

        if return_uint_image:
            return np.clip(image * 255, a_min=0, a_max=255).astype(np.uint8)[::-1]

        return np.clip(image, a_min=0, a_max=1.)


    @staticmethod
    def _filled_white_pixels(image):
        for idx_y in range(image.shape[0]):
            # diff from white color [1,1,1]
            indeces_x = np.where(np.any(image[idx_y, :, :] != [1., 1., 1.], axis=1))[0]
            if len(indeces_x) >= 2:
                min_idx_x, max_idx_x = np.min(indeces_x), np.max(indeces_x)
                for idx_x in range(min_idx_x, max_idx_x + 1):
                    # fill white colors
                    if np.all(image[idx_y, idx_x] == [1,1,1]):
                        window = ImageExtractor.filled_squared_window
                        square = image[idx_y - window:idx_y + window + 1, idx_x - window:idx_x + window + 1, :]
                        indeces_y, indeces_x = np.where(np.any(square[:, :, :] != [1., 1., 1.], axis=(2)))
                        # fill colour if we have not white color
                        if len(indeces_x) > 0:
                            image[idx_y, idx_x] = square[indeces_y, indeces_x, :].mean(axis=(0))
            if idx_y % 10 == 0:
                logging.info(f'Filled idx_y: {idx_y}')
        return image


    @staticmethod
    def _filtered_non_visible_points(list_points, visible_idx_pts):
        list_points_filtered = []
        for pt in list_points:
            # need hash not list (very slow)
            if pt.idx_pt in visible_idx_pts:
                list_points_filtered.append(pt)
        list_points = list_points_filtered
        return list_points


    @staticmethod
    def _get_square_points_correspondence(points_2d, colors_2d, x_range, y_range, bin_mask=False):
        # left_upper point of squared positions (x_0, y_0) : []
        if points_2d.shape[0] != colors_2d.shape[0]:
            raise ValueError

        square_to_points = {}
        found_square_corr = []
        for idx, (pt, color) in enumerate(zip(points_2d, colors_2d)):
            x, y = pt
            right_side_x = np.searchsorted(x_range, x, side='right')
            right_side_y = np.searchsorted(y_range, y, side='right')
            # not in some square
            if right_side_x == 0 or right_side_x == len(x_range):
                continue
            if right_side_y == 0 or right_side_y == len(y_range):
                continue
            # contains in some squared
            square = (right_side_y-1, right_side_x-1)
            # write idx of point also
            point = Point2d(x, y, *color, idx)
            key_square = f'{square[0]}_{square[1]}'
            if key_square in square_to_points:
                square_to_points[key_square].append(point)
            else:
                #print(f'Find corr: {key_square}')
                found_square_corr.append(key_square)
                square_to_points.update({key_square: [point]})
            if idx % 1000 == 0:
                #print(f'Squared correspondence idx: {idx}.')
                pass
        if bin_mask:
            ImageExtractor._visualize_bin_mask_points(found_square_corr, x_range, y_range)
        return square_to_points, found_square_corr


    @staticmethod
    def _visualize_bin_mask_points(found_square_corr, x_range, y_range):
        bin_mask = np.zeros((y_range.shape[0] - 1, x_range.shape[0] - 1, 3), dtype=np.uint8)
        num_found = 0
        for key_square in found_square_corr:
            idx_y, idx_x = key_square.split('_')
            bin_mask[int(idx_y), int(idx_x)] = [255, 255, 255]
            num_found += 1
        print(f'Find: {num_found}')
        cv.imshow('bin_mask', bin_mask[::-1])
        cv.waitKey(0)


    @staticmethod
    def extract_image_not_workming(image_points, rgb_points, shape = (256, 256), interpolate_method=None):
        # we extract image_points and correspondence rgb-points
        if not (isinstance(image_points, np.ndarray) and image_points.shape[1] == 2):
            raise ValueError(f'Shape: {image_points.shape}')
        h, w = shape
        pt_max, pt_min = image_points.max(axis=0), image_points.min(axis=0)
        y_max, x_max = pt_max
        y_min, x_min = pt_min
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        # resize for balance
        w = int(h * (delta_x/delta_y))
        image = np.ones((h, w, 3))
        x_range = np.linspace(x_min - delta_x * 0.1, x_max + delta_x * 0.1, image.shape[1])
        y_range = np.linspace(y_min - delta_y * 0.1, y_max + delta_y * 0.1, image.shape[0])
        dx = delta_x / image.shape[1]
        dy = delta_y / image.shape[0]
        # we need extract image
        number_not_found = 0
        interpolate_pixels = []
        founded_pixels = []
        for idx, x_coor in enumerate(y_range):
            for jdx, y_coor in enumerate(x_range):
                point_mesh = np.array([x_coor, y_coor])
                closest_idx = np.argmin((image_points - point_mesh).sum(axis = 1))
                dist = (np.abs(image_points[closest_idx] - point_mesh)).sum()
                if dist < (dy + dx):
                    image[idx, jdx] = rgb_points[closest_idx]
                    founded_pixels.append(((idx, jdx), rgb_points[closest_idx]))
                    plt.scatter(image_points[:, 0], image_points[:, 1], c='b')
                    plt.show()
                else:
                    number_not_found += 1
                    interpolate_pixels.append([idx, jdx])
            if idx % 10:
                print(f'row: {idx}')
        # if [0,1]
        #image = np.clip(image * 255, a_min=0, a_max=255).astype(np.uint8)
        if interpolate_method:
            print('We use some interpolate method for zeros points')
            pass
        return image


class Point2d():

    def __init__(self, x, y, r, g, b, idx_pt=None):
        self._pt = (x, y)
        self._rgb = (r,g,b)
        self._idx_pt = idx_pt

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

    @staticmethod
    def get_color(list_Points, strategy='mean'):
        if list_Points:
            colors = np.asarray([list(point.color) for point in list_Points])
            if strategy == 'mean':
                return colors.mean(axis=0)
            elif strategy == 'max':
                return colors.max(axis=0)
            elif strategy == 'min':
                return colors.min(axis=0)
        else:
            # white color
            return (1., 1., 1.)