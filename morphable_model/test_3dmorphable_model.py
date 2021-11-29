
import cv2 as cv
import numpy as np
from pathlib import Path
import logging

from face3d.my_folder.morphable_model.utils import MorphableModelFullHead, MeshVisualizer, ImageExtractor
from face3d.my_folder.morphable_model.transformed3d import Transformed3d
from face3d.my_folder.morphable_model.full_head_model import FullHeadModel


format = '%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format)


FULL_HEAD_MODEL_FILE_NAME = Path('/home/mklochkov/projects/data/3dface/model2019_fullHead.h5')
BASEL_FACE_MODEL_FILE_NAME = Path('/home/mklochkov/projects/data/3dface/model2019_bfm.h5')


class Test3DMorphableModel():


    @staticmethod
    def extract_json_structure_basel_model(path_basel_model=BASEL_FACE_MODEL_FILE_NAME):
        bfm = FullHeadModel(path_basel_model)
        root_tree = bfm.json_structure
        print(root_tree)


    @staticmethod
    def generate_random_vertices_texture(visualize=True):
        morph_model = MorphableModelFullHead(BASEL_FACE_MODEL_FILE_NAME)
        mesh_visualizer = MeshVisualizer(morph_model)
        shape_params = morph_model.generate_face_params(type='shape')
        color_params = morph_model.generate_face_params(type='color')
        vertices, texture = morph_model.get_face_vertices_texture(shape_params, color_params)
        triangles = morph_model.triangles
        if visualize:
            mesh_visualizer.visualize_face(vertices, texture, triangles)
        return vertices, texture, triangles


    @staticmethod
    def projected_mean_face_on_image_plane(angle_x=0, angle_y=0, angle_z=0, focal_lenght=1, visible_points=True):
        morph_model = MorphableModelFullHead(BASEL_FACE_MODEL_FILE_NAME)
        mean_shape = morph_model.mean_shape
        mean_color = morph_model.mean_color
        mean_shape_rotate = Transformed3d.rotate3d(mean_shape, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
        projected_points = Transformed3d.get_projection(mean_shape_rotate, focal_lenght=focal_lenght)
        if visible_points:
            visible_idxs = morph_model.get_visible_points(mean_shape_rotate)
        else:
            visible_idxs = None

        projected_points = Transformed3d.get_projection(mean_shape_rotate, focal_lenght=focal_lenght)
        image_mean = ImageExtractor.extract_image_another(projected_points, rgb_points=mean_color,
                                                          return_uint_image=False, visible_idx_pts=visible_idxs)
        image_mean = cv.resize(image_mean, (640, 860))
        image_mean = cv.cvtColor((image_mean * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        cv.imshow('result', image_mean[::-1])
        cv.imwrite('/home/mklochkov/projects/data/3dface/mean_face_rotate.jpg', image_mean[::-1])
        cv.waitKey(0)



if __name__ == '__main__':
    Test3DMorphableModel.extract_json_structure_basel_model(angle_z=10, angle_y=10)
