import matplotlib.pyplot as plt
import open3d
import numpy as np
import cv2 as cv
import logging


from face3d.my_folder.morphable_model.full_head_model import FullHeadModel


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



class MorphableModelFullHead():


    def __init__(self, path_file_model):
        full_head_model = FullHeadModel(path_file_model)
        self.mean_shape = full_head_model.mean_shape
        self.mean_color = full_head_model.mean_color

        self.triangles_shape = full_head_model.repr_cells
        self.shape_points = full_head_model.repr_points

        self.pca_shape = full_head_model.pca_shape
        self.pca_color= full_head_model.pca_color

        self.noiseVariance_shape = full_head_model.noise_variance_shape
        self.noiseVariance_expression = full_head_model.noise_variance_shape
        self.noiseVariance_color = full_head_model.noise_variance_color

        self.pca_shape_variance = full_head_model.pca_shape_variance
        self.pca_color_variance = full_head_model.pca_color_variance

        self.color_repr_cells = full_head_model.color_repr_cells
        self.color_repr_colorspace = full_head_model.color_repr_colorspace
        self.color_repr_points = full_head_model.color_repr_points
        # idx_pt to list_triangles [t_1], t_1 = [t_1^1, t_1^2, t_1^3]
        self._idx_vertex_to_idx_triangles = self._get_connected_triangles()

    # TODO: Have areas where result some bad (ears, and mounth)
    def get_visible_points(self, verteces, camera_unit_vector=np.array([0,0,-1]), strategy='all'):
        # (n_triangles, 3)
        normals = self._get_triangle_normals(verteces)
        assert normals.shape == (self.triangles_shape.shape[0], 3)
        scalar_products = np.dot(normals, camera_unit_vector[:, np.newaxis])
        visible_points = {}
        counter_vis_points = 0
        for idx_pt, idxs_triangles in self._idx_vertex_to_idx_triangles.items():
                # any normal
                if idx_pt % 1000:
                    logging.info(f'Check: {idx_pt} points searching visible points.')
                if strategy == 'one':
                    # (n_normals_connected_pt, scalar_product)
                    if np.any(scalar_products[idxs_triangles] > 0):
                        visible_points[idx_pt] = counter_vis_points
                        counter_vis_points +=1
                elif strategy == 'all':
                    if np.all(scalar_products[idxs_triangles] > 0):
                        visible_points[idx_pt] = counter_vis_points
                        counter_vis_points +=1

                else:
                    raise NotImplemented(f'Not implemented strategy: {strategy} for searching visible points.')
        # just indeces
        return visible_points


    def _get_triangle_normals(self, verteces):
        idx_vertex_to_normals = {}
        # N_triangles, 3, 3
        idxs_1, idxs_2, idxs_3 = self.triangles_shape[:, 0], self.triangles_shape[:, 1], self.triangles_shape[:, 2]
        pts_1, pts_2, pts_3 = verteces[idxs_1], verteces[idxs_2], verteces[idxs_3]
        triangle_normas = self._get_normalize_cross_poduct(pts_1, pts_2, pts_3)
        return triangle_normas


    def _get_normalize_cross_poduct(self, pts_1, pts_2, pts_3):
        cross_product = np.cross(pts_3 - pts_2, pts_2 - pts_1)
        cross_product /= np.linalg.norm(cross_product, axis=1)[:, np.newaxis]
        return cross_product


    def _get_normals_by_one_triangle(self, idx_point, verteces):
        if not (isinstance(verteces, np.ndarray) and verteces.shape[1] == 3):
            raise ValueError('Incorrect verteces array format!')
        triangles = self._idx_vertex_to_idx_triangles[idx_point]
        # define  (p3 - p2) * (p2 - p1)
        normals = np.empty((len(triangles), 3))
        for idx, triangle in enumerate(triangles):
            idx_1, idx_2, idx_3 = triangles
            pt1, pt2, pt3 = verteces[idx_1], verteces[idx_2], verteces[idx_3]
            normal_vector = self._get_normal_by_one_triangle(pt1, pt2, pt3)
            normals[idx] = normal_vector
        return normals

    # very slow if we will use one by one strategy
    def _get_normal_by_one_triangle(self, pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray):
        cross_product = np.cross(pt3 - pt2, pt2 - pt1)
        return cross_product / np.linalg.norm(cross_product)


    def _get_connected_triangles(self):
        idx_vertex_to_idx_triangles = {}
        for idx_triangle, triangle in enumerate(self.triangles_shape):
            for idx_point in triangle:
                self._update_vertex_to_triangles_corr(idx_point, idx_triangle, idx_vertex_to_idx_triangles)
        assert len(idx_vertex_to_idx_triangles) == len(self.mean_shape), 'Number keys in dictionary ' \
                                                                      'should be equal number vetrices!'
        return idx_vertex_to_idx_triangles


    def _update_vertex_to_triangles_corr(self, idx_point, idx_triangle, idx_vertex_to_idxs_triangles):
        if idx_point in idx_vertex_to_idxs_triangles:
            idx_vertex_to_idxs_triangles[idx_point].append(idx_triangle)
        else:
            idx_vertex_to_idxs_triangles[idx_point] = [idx_triangle]


    @property
    def triangles(self):
        return self.triangles_shape


    def generate_face_params(self, type='shape'):
        if type not in ['color', 'shape']:
            raise ValueError
        if type == 'shape':
            number_dims = self.pca_shape_variance.shape[0]
            params = np.random.multivariate_normal(mean = np.zeros(number_dims, ),
                                                   cov = np.diag(self.pca_shape_variance), size=1)[0]
            return params
        elif type == 'color':
            number_dims = self.pca_color_variance.shape[0]
            params = np.random.multivariate_normal(mean = np.zeros(number_dims, ),
                                                   cov=np.diag(self.pca_color_variance), size=1)[0]
            return params
        # TODO: add expression parameters (creation)
        elif type == 'expression':
            assert False


    def get_face_vertices_texture(self, shape_params, color_params, expression=False, check_distr=True):
        if shape_params.shape[0] != 199 and not isinstance(shape_params, np.ndarray):
            raise ValueError('The shape params is wrong!')
        if color_params.shape[0] != 199 and not isinstance(color_params, np.ndarray):
            raise ValueError('The color params is wrong!')
        # shape_params and color_params should be from Gaussian() with some sigmas
        if check_distr:
            shape_params = self._check_parameters_model(shape_params, type='shape')
            color_params = self._check_parameters_model(color_params, type='shape')
        # TODO: How do it with np.dot?
        #generated_shape_vertices = self._mean_shape + np.dot(self._pca_shape, shape_params)
        #generated_texture = self._mean_color + np.dot(self._pca_color, color_params)
        shape_params = np.repeat(shape_params, repeats=3, axis=0).T.reshape((-1, 1, 3))
        color_params = np.repeat(color_params, repeats=3, axis=0).T.reshape((-1, 1, 3))

        generated_shape_vertices = self.mean_shape + np.multiply(self.pca_shape, shape_params).sum(axis = 0)
        generated_texture = self.mean_color + np.multiply(self.pca_color, color_params).sum(axis = 0)

        # TODO: add expression calculation for face
        if expression:
            pass
        return generated_shape_vertices, generated_texture


    def _check_parameters_model(self, params, type='shape', koeff_sigma=2):
        if params.shape[0] != 199 and not isinstance(params, np.ndarray):
            raise ValueError('')
        if type not in ['color', 'shape']:
            raise ValueError('')
        if type == 'shape':
            indeces_out_range = np.where(np.asarray(params) >= koeff_sigma * self.pca_shape_variance)[0]
        else:
            indeces_out_range = np.where(np.asarray(params) >= koeff_sigma * self.pca_color_variance)[0]

        if indeces_out_range:
            params = np.clip(params, a_min=-self.pca_shape_variance * koeff_sigma,
                             a_max=self.pca_shape_variance * koeff_sigma)
        return params



class ImageExtractor():


    @staticmethod
    def extract_image_another(image_points, rgb_points, height_num_pixels=256, return_uint_image=True,
                              visible_idx_pts=None):
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
                list_points_filtered=[]
                for pt in list_points:
                    # need hash not list (very slow)
                    if pt.idx_pt in visible_idx_pts:
                        list_points_filtered.append(pt)
                list_points = list_points_filtered
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
                pass
        coors_found = np.asarray(points_plot)
        if return_uint_image:
            return np.clip(image * 255, a_min=0, a_max=255).astype(np.uint8)[::-1]

        return np.clip(image, a_min=0, a_max=1.)


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
            bin_mask = np.zeros((y_range.shape[0]-1, x_range.shape[0]-1, 3), dtype=np.uint8)
            num_found = 0
            for key_square in found_square_corr:
                idx_y, idx_x = key_square.split('_')
                bin_mask[int(idx_y), int(idx_x)] = [255,255,255]
                num_found += 1
            print(f'Find: {num_found}')
            cv.imshow('bin_mask', bin_mask[::-1])
            cv.waitKey(0)
        return square_to_points, found_square_corr


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



class WrapperOpen3dType():


    @staticmethod
    def get_open3d_obj(tensor, type='float', vectors3d=False):
        if not isinstance(tensor, (np.ndarray, list)):
            raise ValueError
        if vectors3d:
            return open3d.utility.Vector3dVector(tensor)
        else:
            if type=='float':
                type_open3d = open3d.core.Dtype.Float32
            else:
                type_open3d = open3d.core.Dtype.Int32
            return open3d.core.Tensor(tensor, type_open3d)

    @staticmethod
    def get_vector(tensor, type='float'):
        if not isinstance(tensor, (np.ndarray, list)):
            raise ValueError
        if type not in ['float', 'int']:
            raise ValueError
        if type == 'float':
            return open3d.utility.Vector3dVector(tensor)
        # for triangles
        elif type == 'int':
            return open3d.utility.Vector3iVector(tensor)



class MeshVisualizer():


    def __init__(self, morphable_head_model: MorphableModelFullHead):
        self._morphable_head_model = morphable_head_model


    def visualize_face(self, vertices, texture, triangles, width=640, height=480):
        if not isinstance(vertices, np.ndarray) and isinstance(texture, np.ndarray)\
                and isinstance(triangles, np.ndarray):
            raise ValueError('Not np.ndarrays!')
        face_mesh = open3d.geometry.TriangleMesh()
        face_mesh.vertices = WrapperOpen3dType.get_vector(vertices, type='float')
        face_mesh.triangles = WrapperOpen3dType.get_vector(triangles, type='int')
        face_mesh.vertex_colors = WrapperOpen3dType.get_vector(texture, 'float')
        open3d.visualization.draw_geometries([face_mesh], width=width, height=height)


    def visualize_mean_shape(self, height=480, width=640):
        mean_shape_pts = self._morphable_head_model.mean_shape
        colors_mean_shape = self._morphable_head_model.mean_color
        triangles_shape = self._morphable_head_model.triangles_shape
        face_mesh = open3d.geometry.TriangleMesh()
        face_mesh.vertices = WrapperOpen3dType.get_vector(mean_shape_pts, type='float')
        face_mesh.triangles = WrapperOpen3dType.get_vector(triangles_shape, type='int')
        face_mesh.vertex_colors = WrapperOpen3dType.get_vector(colors_mean_shape, 'float')
        open3d.visualization.draw_geometries([face_mesh], width=width, height=height)


    # is not a shape
    def visualize_one_of_the_pca(self, index_pca_component, mesh = False, height=480, width=640):
        shape_pca = self._morphable_head_model.pca_shape[index_pca_component]
        color_pca = self._morphable_head_model.pca_color[index_pca_component]
        triangles = self._morphable_head_model.triangles_shape
        if mesh:
            mesh = open3d.geometry.TriangleMesh()
            mesh.vertices = WrapperOpen3dType.get_vector(shape_pca, type='float')
            mesh.triangles = WrapperOpen3dType.get_vector(triangles, type='int')
            mesh.vertex_colors = WrapperOpen3dType.get_vector(color_pca, 'float')
            open3d.visualization.draw_geometries([mesh], width=width, height=height)
        else:
            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = WrapperOpen3dType.get_vector(shape_pca, type='float')
            open3d.visualization.draw_geometries([point_cloud], width=width, height=height)
