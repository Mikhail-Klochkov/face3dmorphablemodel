from pathlib import Path
import h5py
import numpy as np
from face3d.my_folder.morphable_model.tree_h5_file import TreeH5file
from face3d.mesh_numpy.vis import plot_mesh


FULL_HEAD_MODEL_FILE_NAME = Path('/home/mklochkov/projects/data/3dface/model2019_fullHead.h5')
BASEL_FACE_MODEL_FILE_NAME = Path('/home/mklochkov/projects/data/3dface/model2019_bfm.h5')



class FullHeadModel(object):


    def __init__(self, path_file):
        if isinstance(path_file, str):
            path_file = Path(path_file)
        if path_file.is_file():
            self._path_file = path_file
        else:
            assert False

    @property
    def name_groups(self):
        with h5py.File(self._path_file, 'r') as reader:
            keys = list(reader.keys())
        return keys


    @property
    def json_structure(self, debug = True):
        tree_structure_h5file = TreeH5file(self._path_file)
        tree_structure_h5file.build_tree_struct()
        if debug:
            tree_structure_h5file.show_tree_structure()
        return tree_structure_h5file.tree_structure_data


    def _get_dataset(self, key, processed_f):
        with h5py.File(self._path_file, 'r') as reader:
            try:
                if processed_f:
                    return processed_f(reader[key])
                else:
                    assert False, 'processed_f is None!'
            except Exception as e:
                assert False, f'Error reading file! The error type: {e}'

    @property
    def mean_shape(self):
        key = 'shape/model/mean'
        def _processed_mean_shape(mean):
            return np.asarray(mean, dtype=np.float32).reshape((-1, 3))
        return self._get_dataset(key, processed_f= _processed_mean_shape)


    @property
    def mean_color(self):
        key = 'color/model/mean'

        def _processed_mean_color(mean):
            return np.asarray(mean, dtype=np.float32).reshape((-1, 3))

        return self._get_dataset(key, processed_f=_processed_mean_color)


    @property
    def pca_shape_variance(self):
        key = 'shape/model/pcaVariance'

        def _processed_pca_shape_variance(pca_shape_variance):
            return np.asarray(pca_shape_variance)
        return self._get_dataset(key, processed_f=_processed_pca_shape_variance)


    @property
    def pca_color_variance(self):
        key = 'color/model/pcaVariance'

        def _processed_pca_color_variance(pca_color_variance):
            return np.asarray(pca_color_variance)
        return self._get_dataset(key, processed_f=_processed_pca_color_variance)


    @property
    def repr_cells(self):
        key = 'shape/representer/cells'

        def _processed_shape_cells(cells):
            return np.asarray(cells, dtype=np.int32).T

        return self._get_dataset(key, processed_f=_processed_shape_cells)


    @property
    def repr_points(self):
        key = 'shape/representer/points'

        def _processed_shape_points(cells):
            return np.asarray(cells, dtype=np.float32)

        return self._get_dataset(key, processed_f=_processed_shape_points)


    @property
    def pca_shape(self):
        key = 'shape/model/pcaBasis'
        def _processed_pca_shape(pca_shape):
            return np.asarray(pca_shape, dtype=np.float32).reshape((199, -1, 3))

        return self._get_dataset(key, processed_f=_processed_pca_shape)


    @property
    def noise_variance_shape(self):
        return self._noise_variance(type='shape')


    @property
    def noise_variance_expression(self):
        return self._noise_variance(type='expression')


    @property
    def noise_variance_color(self):
        return self._noise_variance(type='color')


    def _noise_variance(self, type):
        assert type in ['color', 'expression', 'shape']
        key = f'{type}/model/noiseVariance'
        def _preprocessed_noiseVariance(noiseVariance, type=None):
            return np.asarray(noiseVariance, dtype=np.float32)

        return self._get_dataset(key, processed_f=_preprocessed_noiseVariance)


    @property
    def pca_color(self):
        key = 'color/model/pcaBasis'

        def _processed_pca_color(pca_color):
            return np.asarray(pca_color, dtype=np.float32).reshape((199, -1, 3))

        return self._get_dataset(key, processed_f=_processed_pca_color)

    @property
    def color_repr_cells(self):
        key = 'color/representer/cells'

        def _processed_color_repr_cells(color_repr_cells):
            return np.asarray(color_repr_cells)

        return self._get_dataset(key, processed_f=_processed_color_repr_cells)

    @property
    def color_repr_colorspace(self):
        key = 'color/representer/colorspace'

        def _processed_color_repr_colorspace(color_repr_colorspace):
            return np.asarray(color_repr_colorspace)

        return self._get_dataset(key, processed_f=_processed_color_repr_colorspace)

    @property
    def color_repr_points(self):
        key = 'color/representer/points'

        def _processed_color_repr_points(color_repr_points):
            return np.asarray(color_repr_points)

        return self._get_dataset(key, processed_f=_processed_color_repr_points)

    # in another block
    def plot_face(self, subplot=[1,1,1], title = 'mesh', el = 90, az = -90,
                  lwdt=.1, dist = 6, color="grey"):
        vertices = self.mean_shape
        triangles = self.repr_cells
        vertices, triangles = self.eliminate_face_pts(vertices, triangles, False)
        plot_mesh(vertices, triangles, subplot, title, el, az, lwdt, dist, color)


    def eliminate_face_pts(self, vertices, triangles, not_removed=True):
        # remove some pts
        if not_removed:
            vertices_new = vertices.copy()
            triangles_new = triangles.copy()
            return vertices_new, triangles_new
        else:
            raise ValueError('Not removed should be True!')

