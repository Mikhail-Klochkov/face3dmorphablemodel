
import cv2 as cv
import open3d
import numpy as np
from pathlib import Path
import logging
import scipy.io
import matplotlib.pyplot as plt


from sklearn.metrics import pairwise_distances


from face3d.my_folder.morphable_model.mesh_visualizer import WrapperOpen3dType
from face3d.my_folder.morphable_model.morphable_model_full_head import MorphableModelFullHead
from face3d.my_folder.morphable_model.mesh_visualizer import MeshVisualizer
from face3d.my_folder.morphable_model.image_extractor import ImageExtractor
from face3d.my_folder.morphable_model.transformed3d import Transformed3d
from face3d.my_folder.morphable_model.full_head_model import FullHeadModel
from face3d.my_folder.morphable_model.keypoints_extractor import KeyPointsExtractor, FaceDetectorWrapper
from face3d.my_folder.morphable_model.face_detector_dlib import FaceRecognizerDlib
from face3d.my_folder.morphable_model.point_correspond import PointCorresponder



format = '%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format)


FULL_HEAD_MODEL_FILE_NAME_2019 = Path('/home/mklochkov/projects/data/3dface/model2019_fullHead.h5')
BASEL_FACE_MODEL_FILE_2019 = Path('/home/mklochkov/projects/data/3dface/model2019_bfm.h5')
BASEL_FACE_MODEL_FILE_2019_nomouth = Path('/home/mklochkov/projects/data/3dface/model2019_face12.h5')

BASEL_FACE_MODEL_FILE_2017_h5 = Path('/home/mklochkov/projects/data/3dface/model2017-1_bfm_nomouth.h5')
BASEL_FACE_MODEL_FILE_2017_nomouth_h5 = Path('/home/mklochkov/projects/data/3dface/model2017-1_face12_nomouth.h5')


BASEL_FACE_MODLE_FILE_2017 = Path('/home/mklochkov/projects/data/3dface/Basel2017model.mat')
BASEL_FACE_REGIONS_2017 = Path('/home/mklochkov/projects/data/3dface/face05_4seg.txt')
BASEL_FACE_MODEL_FILE_2019_update = Path('/home/mklochkov/projects/data/3dface/model2019_face12.h5')

path_dir_dlib_models = Path('/home/mklochkov/projects/data/models')


class Test3DMorphableModel():


    @staticmethod
    def extract_json_structure_basel_model(path_basel_model):
        bfm = FullHeadModel(path_basel_model)
        root_tree = bfm.json_structure
        print(root_tree)

    @staticmethod
    def visualize_mean_shape_pca(path_model=BASEL_FACE_MODEL_FILE_2019):
        morph_model = MorphableModelFullHead(path_model)
        print(morph_model.mean_shape.shape)
        mesh_visualizer = MeshVisualizer(morph_model)
        mesh_visualizer.visualize_mean_shape_point_cloud()


    @staticmethod
    def visualize_random_face(path_model=BASEL_FACE_MODEL_FILE_2019):
        morph_model = MorphableModelFullHead(path_model)
        random_shape = morph_model.generate_shape_random_face()
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(random_shape)
        open3d.visualization.draw_geometries([pcd], width=800, height=1000)


    @staticmethod
    def visualize_mean_shape():
        morph_model = MorphableModelFullHead(BASEL_FACE_MODEL_FILE_2019)
        mesh_visualizer = MeshVisualizer(morph_model)
        mesh_visualizer.visualize_mean_shape()


    @staticmethod
    def generate_random_vertices_texture(face_model_file=BASEL_FACE_MODEL_FILE_2019, visualize=True):
        morph_model = MorphableModelFullHead(face_model_file)
        mesh_visualizer = MeshVisualizer(morph_model)
        shape_params = morph_model.generate_face_params(type='shape')
        color_params = morph_model.generate_face_params(type='color')
        vertices, texture = morph_model.get_face_vertices_texture(shape_params, color_params)
        triangles = morph_model.triangles
        if visualize:
            mesh_visualizer.visualize_face(vertices, texture, triangles)
        return vertices, texture, triangles


    @staticmethod
    def projected_mean_face_on_image_plane(angle_x=0, angle_y=0, angle_z=0, focal_lenght=1, stay_visible_pts=True,
                                           visualize=True, resize=False, use_z_coors=False):
        morph_model = MorphableModelFullHead(BASEL_FACE_MODEL_FILE_2019)
        mean_shape = morph_model.mean_shape
        mean_color = morph_model.mean_color
        mean_shape_rotate = Transformed3d.rotate3d(mean_shape, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
        if stay_visible_pts:
            # idx point from mean_shape to idx of visible_point
            visible_idxs = morph_model.get_visible_points(mean_shape_rotate)
        else:
            visible_idxs = None
        # TODO: need to save also z coordidnate for selection point on image plane
        projected_points, z_coors = Transformed3d.get_projection(mean_shape_rotate, focal_lenght=focal_lenght,
                                                        return_z_coor=use_z_coors)
        if z_coors is None:
            z_coors = None
        image_mean, pixel_to_pt = ImageExtractor.extract_image_another(projected_points,
                                                                       rgb_points=mean_color,
                                                                       return_uint_image=False,
                                                                       visible_idx_pts=visible_idxs,
                                                                       z_coors=z_coors)
        if resize:
            image_mean = cv.resize(image_mean, (640, 860))
        image_mean = cv.cvtColor((image_mean * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        if visualize:
            cv.imshow('result', image_mean[::-1])
            cv.imwrite('/home/mklochkov/projects/data/3dface/mean_face_rotate.jpg', image_mean[::-1])
            cv.waitKey(0)
        # get image, 2d points of mean_shape, indeces_which visible
        return image_mean, projected_points, visible_idxs, pixel_to_pt


    @staticmethod
    def extract_keypoints_stats_mean(number_generated_samples=10, use_z_coors=False, visualization=False):
        # define possible angles (uniform)
        # atitude of variations of angles (x, y, z)
        angle_b = 5
        keypoints_extractor_new = FaceDetectorWrapper()
        keypoints_extractor_new.prepare(ctx_id=0, det_size=(640, 640))
        dlibfacerecognizer = FaceRecognizerDlib(path_dir_dlib_models)
        for it in range(number_generated_samples):
            angles = []
            for axis in ['x', 'y', 'z']:
                angles += [generate_angle_uniform(-angle_b, angle_b)]
            angle_x, angle_y, angle_z = angles
            print(f'generate angles: {angle_x}, {angle_y}, {angle_z}')
            # generate image
            mean_sample_img, projected_pts, \
            indeces_visible, pixel_to_pt = Test3DMorphableModel.projected_mean_face_on_image_plane(angle_x, angle_y,
                                                        angle_z, use_z_coors=use_z_coors, resize=False, visualize=False)
            # vertical flip image
            mean_sample_img = mean_sample_img[::-1]
            scores, boxes, keypoints_small = keypoints_extractor_new.detect_faces_wrapper(mean_sample_img)
            try:
                box = boxes[0]
            except Exception as e:
                logging.info(f'Face not found! Iteration: {it}. Angles: {[angle_x, angle_y, angle_z]}')
                continue
            face_landmarks = dlibfacerecognizer.get_face_landmarks(mean_sample_img, box)
            keypoints_big = np.asarray([[pt.x, pt.y] for pt in face_landmarks.parts()])
            # TODO: Need change code (Wrong projected_pts)
            if visualization:
                idx_to_pt = {idx: pt for idx, pt in enumerate(pixel_to_pt.values())}
                idx_to_pixel = {idx: pixel for idx, pixel in enumerate(pixel_to_pt.keys())}
                #projected_pts_arr = np.asarray([list(pt._pt) for pixel, pt in pixel_to_pt.items()])
                projected_pts_arr = np.asarray([get_coor_by_pixel(pixel) for pixel, pt in pixel_to_pt.items()])
                # need transfrom data (need rotate and shift by some vector)
                pts_centered, mean_pt = center_data(projected_pts_arr)
                pts_centered_rot = rotate_pts_angle(pts_centered, angle=90)
                pts_centered_rot += mean_pt
                distances = pairwise_distances(keypoints_big, pts_centered_rot, metric='euclidean')
                #distances = pairwise_distances(keypoints_big, projected_pts_arr, metric='euclidean')
                closest_idxs = np.argsort(distances, axis=1)[:, 0]
                closest_pts = [idx_to_pt[idx] for idx in closest_idxs]
                closest_pixels = [pts_centered_rot[idx] for idx in closest_idxs]
                #closest_pixels = [idx_to_pixel[idx] for idx in closest_idxs]
                mean_sample_img_copy = np.clip(mean_sample_img.astype(np.float32).copy() / 255, a_min=0, a_max=1.)
                for key_pt, closest_pixel in zip(keypoints_big, closest_pixels):
                    x, y = key_pt
                    mean_sample_img_copy = cv.circle(mean_sample_img_copy, (x, y), radius=2, color=(0, 1, 0), thickness=-1)
                    y_pixel, x_pixel = closest_pixel
                    y_pixel, x_pixel = int(y_pixel), int(x_pixel)
                    mean_sample_img_copy = cv.circle(mean_sample_img_copy, (y_pixel, x_pixel), radius=1, color=(1, 0, 0),
                                                     thickness=-1)
                mean_sample_img_copy = cv.resize(mean_sample_img_copy, (640, 860))
                cv.imshow('win', mean_sample_img_copy)
                cv.waitKey(0)
                mean_shape = MorphableModelFullHead(BASEL_FACE_MODEL_FILE_2019).mean_shape
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(mean_shape)
                pcd.colors = open3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(mean_shape))])
                pcd_landmarks = open3d.geometry.PointCloud()
                indeces_3d_pts = [pt.idx_pt for pt in closest_pts]
                landmarks = np.asarray([mean_shape[id] for id in indeces_3d_pts])
                pcd_landmarks.points = open3d.utility.Vector3dVector(landmarks)
                pcd_landmarks.voxel_down_sample(voxel_size=3)
                pcd_landmarks.colors = open3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(mean_shape))])
                open3d.visualization.draw_geometries([pcd, pcd_landmarks], width=800, height=1000)


def get_rotate_matr(angle=90):
    rad = np.radians(angle)
    return np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])


def rotate_pts(pts, rot):
    return np.dot(rot, pts.T).T


def center_data(pts, inverse_mean_pt=True, const_shift=15):
    mean_pt = pts.mean(axis=0)
    if inverse_mean_pt:
        mean_inv = mean_pt[::-1]
        mean_inv[1] -= const_shift
        return pts - mean_pt, mean_inv
    else:
        return pts - mean_pt, mean_pt


def rotate_pts_angle(pts, angle=90):
    return rotate_pts(pts, get_rotate_matr(angle))


def get_coor_by_pixel(pixel):
    y_pixel, x_pixel = pixel.split('_')
    y_pixel, x_pixel = int(y_pixel), int(x_pixel)
    return y_pixel, x_pixel


def generate_angle_uniform(a, b):
    return np.random.random() * (b - a) + a


def check_basel_2017(file=BASEL_FACE_MODLE_FILE_2017):
    basel_face_model_2017 = scipy.io.loadmat(str(file))
    print(type(basel_face_model_2017))


def _extract_mask_regions(file_regions=BASEL_FACE_REGIONS_2017):
    with BASEL_FACE_REGIONS_2017.open('r') as reader:
        regions = reader.read().split('\n')[:-1]
        mask_regions = np.asarray([int(id_region) for id_region in regions], dtype=np.int)
    return mask_regions


def _smallest_right_value(arr, idx):
    return np.searchsorted(arr, idx, side='right')


def _get_current_indeces(indeces_removed, number_verteces):
    # indeces_removed need be sorted
    if not np.all(np.diff(indeces_removed) >= 0):
        logging.info(f'Need sorted Indeced_removed!')
        indeces_removed = sorted(indeces_removed)
    Indeces_now = []
    for idx in np.arange(number_verteces):
        if idx in indeces_removed:
            Indeces_now += [-1]
        else:
            smallest_idx_r = _smallest_right_value(indeces_removed, idx)
            Indeces_now += [idx - smallest_idx_r]
    assert number_verteces == len(Indeces_now)
    return Indeces_now


def _remove_regions_from_mean_face(file_basel_model=BASEL_FACE_MODLE_FILE_2017, file_regions=BASEL_FACE_REGIONS_2017,
                                   regions=4, triangles_remove=True):
    basel_face_model_2017 = scipy.io.loadmat(str(file_basel_model))
    print(f'Keys: {basel_face_model_2017.keys()}')
    shape_mean = basel_face_model_2017['shapeMU'].reshape((-1, 3))
    color_mean = basel_face_model_2017['texMU'].reshape((-1, 3))
    triangles = basel_face_model_2017['tl']
    if regions is not None:
        mask_regions = _extract_mask_regions(file_regions)
        indeces_without_region = np.where(mask_regions!=regions)[0]
        indeces_region = np.where(mask_regions==regions)[0]
        indeces_now = _get_current_indeces(indeces_region, shape_mean.shape[0])
        shape_mean = shape_mean[indeces_without_region]
        color_mean = color_mean[indeces_without_region]
        mask_triangles_stay = []
        if triangles_remove:
            for idx_t, t in enumerate(triangles):
                # triangles indeces from 1...NV
                if any([indeces_now[i-1] == -1 for i in t]):
                    # need remove this string
                    mask_triangles_stay += [False]
                else:
                    # triangles indeces from 1...NV
                    triangles[idx_t] = [indeces_now[i-1] for i in t]
                    mask_triangles_stay += [True]
            triangles = triangles[mask_triangles_stay]
    return shape_mean, color_mean, triangles


def _visualize_mean_2017(shape_mean, color_mean, triangles, height=820, width=1000):
    face_mesh = open3d.geometry.TriangleMesh()
    face_mesh.vertices = WrapperOpen3dType.get_vector(shape_mean, type='float')
    face_mesh.triangles = WrapperOpen3dType.get_vector(triangles, type='int')
    face_mesh.vertex_colors = WrapperOpen3dType.get_vector(np.clip(color_mean/255., 0, 1), 'float')
    open3d.visualization.draw_geometries([face_mesh], width=width, height=height)


def visualize_mean_face_without_region_2017():
    mean_shape, mean_color, triangles = _remove_regions_from_mean_face(triangles_remove=True)
    _visualize_mean_2017(mean_shape, mean_color, triangles)


if __name__ == '__main__':
    #Test3DMorphableModel.projected_mean_face_on_image_plane(angle_z=10, angle_y=3, angle_x=3)
    #Test3DMorphableModel.generate_random_vertices_texture()
    #Test3DMorphableModel.visualize_mean_shape()
    #visualize_mean_face_without_region_2017()
    #Test3DMorphableModel.extract_json_structure_basel_model(BASEL_FACE_MODEL_FILE_2017_h5)
    #Test3DMorphableModel.visualize_mean_shape_pcd(BASEL_FACE_MODEL_FILE_2019)
    Test3DMorphableModel.extract_keypoints_stats_mean(use_z_coors=True, visualization=True)
    #Test3DMorphableModel.extract_json_structure_basel_model(BASEL_FACE_MODEL_FILE_2019)
    #Test3DMorphableModel.visualize_random_face()