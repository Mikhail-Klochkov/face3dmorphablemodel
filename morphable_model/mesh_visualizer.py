import numpy as np
import open3d


class MeshVisualizer():


    def __init__(self, morphable_head_model):
        self._morphable_head_model = morphable_head_model


    def visualize_face(self, vertices, texture, triangles, width=820, height=640):
        if not isinstance(vertices, np.ndarray) and isinstance(texture, np.ndarray)\
                and isinstance(triangles, np.ndarray):
            raise ValueError('Not np.ndarrays!')
        face_mesh = open3d.geometry.TriangleMesh()
        face_mesh.vertices = WrapperOpen3dType.get_vector(vertices, type='float')
        face_mesh.triangles = WrapperOpen3dType.get_vector(triangles, type='int')
        face_mesh.vertex_colors = WrapperOpen3dType.get_vector(texture, 'float')
        open3d.visualization.draw_geometries([face_mesh], width=width, height=height)


    def visualize_mean_shape_point_cloud(self, height=820, width=1000):
        mean_shape_pts = self._morphable_head_model.mean_shape
        colors_mean_shape = self._morphable_head_model.mean_color
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(mean_shape_pts)
        pcd.colors = open3d.utility.Vector3dVector(colors_mean_shape)
        open3d.visualization.draw_geometries([pcd], width=width, height=height)


    def visualize_mean_shape(self, height=820, width=1000, add_keypoints=False):
        mean_shape_pts = self._morphable_head_model.mean_shape
        colors_mean_shape = self._morphable_head_model.mean_color
        triangles_shape = self._morphable_head_model.triangles_shape
        if add_keypoints:
            add_shape_points = self._morphable_head_model.extract_landmarks_coordinates()
            add_colors_shape = np.full((add_shape_points.shape[0], 3), [0., 1., 0.])
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(add_shape_points)
            pcd.colors = open3d.utility.Vector3dVector(add_colors_shape)
            mean_shape_pts = np.vstack([mean_shape_pts, add_shape_points])
            colors_mean_shape = np.vstack([colors_mean_shape, add_colors_shape])

        face_mesh = open3d.geometry.TriangleMesh()
        face_mesh.vertices = WrapperOpen3dType.get_vector(mean_shape_pts, type='float')
        face_mesh.triangles = WrapperOpen3dType.get_vector(triangles_shape, type='int')
        face_mesh.vertex_colors = WrapperOpen3dType.get_vector(colors_mean_shape, 'float')
        if add_keypoints:
            open3d.visualization.draw_geometries([face_mesh, pcd], width=width, height=height)
        else:
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