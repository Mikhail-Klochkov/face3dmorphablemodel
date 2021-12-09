import numpy as np
from scipy.spatial.transform import Rotation as R


class Transformed3d():


    @staticmethod
    def rotate3d(vectors, angle_x, angle_y, angle_z, rad=False):
        if not isinstance(vectors, np.ndarray):
            raise TypeError
        if len(vectors.shape) == 2:
            if vectors.shape[1] != 3:
                ValueError(f'Vector should be have 3 dim! But was taken: {vectors.shape}.')
        elif len(vectors.shape) == 1:
            if vectors.shape[0] != 3:
                raise ValueError('Vector should be have 3 dim! But was taken: {vector.shape}.')
        else:
            ValueError(f'vector is tensor of shape: {vectors.shape}')
        if not rad:
            rad_x, rad_y, rad_z = np.radians([angle_x, angle_y, angle_z])
        else:
            rad_x, rad_y, rad_z = angle_x, angle_y, angle_z
        return Transformed3d._rotate(vectors, rad_x, rad_y, rad_z, scaling=1).T


    @staticmethod
    def _rotate(vector, rad_x, rad_y, rad_z, scaling=1):
        rotation_matrix = R.from_quat([rad_x, rad_y, rad_z, scaling])
        return np.dot(rotation_matrix.as_matrix(), vector.T)


    @staticmethod
    def translate(vectors, translate_vect):
        if not (isinstance(vectors, np.ndarray) and isinstance(translate_vect, np.ndarray)):
            raise TypeError
        return vectors + translate_vect


    @staticmethod
    def rigit_transformation(vectors, angle_x, angle_y, angle_z, translate_vec):
        rotated_vectors = Transformed3d.rotate3d(vectors, angle_x, angle_y, angle_z)
        return Transformed3d.translate(rotated_vectors, translate_vec)


    @staticmethod
    def get_projection(vectors_camera_coor, focal_lenght, principle_point = (0, 0), scaling_factor=1,
                       return_z_coor=True):
        o_x, o_y = principle_point
        # principle point can be (h/2, w/2)
        K_intristic = np.array([[focal_lenght, 0, o_x], [0, focal_lenght, o_y]], dtype=np.float32)
        # vector points
        if vectors_camera_coor.shape[0] != K_intristic.shape[1]:
            vectors_camera_coor = vectors_camera_coor.T
        if return_z_coor:
            z_coors = vectors_camera_coor[2:, :].flatten()
            return (scaling_factor * np.dot(K_intristic, vectors_camera_coor)).T, z_coors
        else:
            return (scaling_factor * np.dot(K_intristic, vectors_camera_coor)).T, None