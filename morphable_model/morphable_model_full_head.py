import numpy as np
import logging


from face3d.my_folder.morphable_model.full_head_model import FullHeadModel
from face3d.my_folder.morphable_model.keypoints_extractor import KeyPointsExtractor


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


    def get_corr_keypts_pts_to_vertex_model(self, synthesized_image, name_method='insightface'):
        pass

