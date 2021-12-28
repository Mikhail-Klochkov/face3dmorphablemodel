import logging
import scipy.io
import random
from pathlib import Path
import open3d
import re
import cv2 as cv
import numpy as np
from plyfile import PlyData
from sklearn.decomposition import FastICA, fastica


from face3d.my_folder.morphable_model.mesh_visualizer import WrapperOpen3dType


class Dataset():


    def __init__(self, path_dir):
        self.path_dir = self.check_path(path_dir)


    def check_path(self, path):
        if isinstance(path, str):
            path = Path(path)
        if not (path.is_dir() or path.is_file()):
            raise FileNotFoundError(f'File not found: {path}.')
        return path


class D3dfaceScans(Dataset):


    def __init__(self, path_directory, path_face_scans_paths=None):
        super().__init__(path_directory)
        if path_face_scans_paths:
            path_face_scans_paths = self.check_path(path_face_scans_paths)
            logging.info(f'We find mapping file!')
            self.person_names_to_paths = self.load_mapping_person_names_to_paths(path_face_scans_paths)
        else:
            logging.info(f'We not found mapping file! Start building mapping file!')
            self.person_names_to_paths = self.build_mapping_person_names_to_paths(path_face_scans_paths)


    def build_mapping_person_names_to_paths(self, file_path=None, num_extracted_faces_per_pers=100):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path is None:
            file_path = self.path_dir / 'mapping_persons_paths.txt'
        person_names_to_paths = {}
        with file_path.open('w') as writer:
            for person_name in self.get_person_names():
                writer.write(f'{person_name}\n')
                selected_paths = self.extract_random_paths_scans(person_name, num_extracted_faces_per_pers)
                person_names_to_paths[person_name] = selected_paths
                for idx, path in enumerate(selected_paths):
                    writer.write(f'{idx} {str(path)}\n')

        return person_names_to_paths


    def load_mapping_person_names_to_paths(self, file_path):
        file_path = self.check_path(file_path)
        person_names_to_paths = {}
        with file_path.open('r') as reader:
            current_person_name = None
            for idx_line, line in enumerate(reader):
                logging.info(f'Extracted {idx_line} lines from file mapping!')
                if line.find('/') == -1:
                    current_person_name = line.strip(' \n')
                    person_names_to_paths[current_person_name] = []
                else:
                    idx_path, path = line.strip('\n').split(' ')
                    person_names_to_paths[current_person_name].append(Path(path.strip(' \n')))

        return person_names_to_paths


    def get_person_names(self):
        person_folders = list(self.path_dir.rglob('*/'))
        person_folders = [person_folder for person_folder in person_folders if re.search(r'[0-9]', person_folder.stem) is None]
        person_folders = [path.stem for path in person_folders if path.is_dir()]
        return person_folders


    def perform_ICA(self, n_components=200, number_extracted_scans_per_face=2, random_state=42, matrix_extract=False):
        if matrix_extract:
            X_scans = self.create_data_matrix(number_extracted_scans_per_face, visualize=False)
        else:
            X_scans = self.load_matrix_by_path()
        fast_ica = FastICA(n_components, random_state=random_state)
        print('Data Shape: ', X_scans.shape)
        #X_scans = X_scans.reshape((X_scans.shape[0], -1))
        #X_scans_new = fast_ica.fit_transform(X_scans)
        X_scans = X_scans.reshape((X_scans.shape[0], -1))
        K, W, S = fastica(X_scans, n_components)
        print('Data Shape New: ', K.shape, W.shape, S.shape)


    def load_matrix_by_path(self, path=None):
        if path is None:
            acceptable_paths = [path for path in self.path_dir.rglob('*.npy')]
            number_elements = [int(p.stem.split('_')[-1]) for p in acceptable_paths]
            idx_max = np.argmax(np.asarray(number_elements))
            path = acceptable_paths[idx_max]
            logging.info(f'Loading scan matrix witj number of elements: {number_elements[idx_max]}! Path: {path}.')
        path = self.check_path(path)
        with path.open('rb') as reader:
            data_matrix = np.load(reader)
        return data_matrix


    def create_data_matrix(self, number_samples_per_person=100, shuffle=True, visualize=True, save_matrix=True):
        person_names = self.get_person_names()
        X_scans = []
        idx_to_special_id_face = {}
        for person_name in person_names:
            logging.info(f'Extracted new person face scans: {person_name}!')
            if not hasattr(self, 'person_names_to_paths'):
                selected_paths = self.extract_random_paths_scans(person_name, number_samples_per_person)
            else:
                selected_paths = self.person_names_to_paths[person_name]
            for idx_face, path in enumerate(selected_paths):
                face_scan = self.extract_scan(path)
                sub_folder_specific_name = path.parent.stem
                abbr = sub_folder_specific_name
                specific_abbr_name = [num for num in abbr.split('+')]
                triangles, verteces = self.get_verteces_and_triangles(face_scan)
                if idx_face == 0:
                    if visualize:
                        self.visualize_face_mesh(triangles, verteces)
                idx = len(X_scans)
                idx_to_special_id_face[idx] = person_name + '_' + '_'.join(specific_abbr_name)
                X_scans.append(verteces)
        if shuffle:
            random.shuffle(X_scans)

        X_scans_data = np.asarray(X_scans)
        if save_matrix:
            logging.info(f'Saving data matrix with {len(X_scans_data)} elements!')
            path_matrix = self.path_dir / f'data_matrix_{len(X_scans)}.npy'
            with path_matrix.open('wb') as writer:
                np.save(writer, X_scans_data)

        return X_scans_data


    def extract_random_paths_scans(self, person_name, number_face_scans=100):
        all_paths = self.extract_paths_all_faces_per_person(person_name)
        if number_face_scans >= len(all_paths):
            number_face_scans = len(all_paths)
        logging.info(f'Number scans: {len(all_paths)} for person: {person_name}!')
        selected_paths = random.choices(all_paths, k=number_face_scans)
        return selected_paths


    def extract_paths_all_faces_per_person(self, person_name):
        path_person_folder = self.path_dir / person_name
        path_person_folder = self.check_path(path_person_folder)
        path_all_scans = []
        for sub_path_per_person in path_person_folder.rglob('*/'):
            path_all_scans += [path for path in sub_path_per_person.rglob('*.ply')]
        return path_all_scans


    def extract_3d_scans_sequentially(self, person_name, visualize=True):
        path_dir_person = self.path_dir / person_name
        path_dir_person = self.check_path(path_dir_person)
        for idx_sub_dir, sub_dir in enumerate(path_dir_person.rglob('')):
            for idx_face_scan, face_scan in enumerate(self.extract_3d_scans_per_sub_folder(sub_dir)):
                triangle, verteces = self.get_verteces_and_triangles(face_scan)
                if visualize:
                    self.visualize_face_mesh(triangle, verteces)
                yield triangle, verteces, sub_dir


    def visualize_face_mesh(self, triangle, verteces):
        face_mesh = open3d.geometry.TriangleMesh()
        face_mesh.vertices = WrapperOpen3dType.get_vector(verteces, type='float')
        face_mesh.triangles = WrapperOpen3dType.get_vector(triangle, type='int')
        face_mesh.compute_vertex_normals()
        open3d.visualization.draw_geometries([face_mesh], width=800, height=1000)


    def get_verteces_and_triangles(self, face_scan):
        if not ('vertex' in face_scan and 'face' in face_scan):
            raise KeyError(f'Data scan not contain keys: ["vertex", "face"]')
        verteces, triangles_face = face_scan['vertex'].data, face_scan['face'].data
        verteces = np.asarray(verteces.data)
        verteces = np.asarray([list(row) for row in verteces])
        triangle = np.asarray(triangles_face.data)
        triangle = np.asarray([list(row) for row in triangle]).astype(np.int32)
        triangle = triangle.reshape((-1, 3))
        return triangle, verteces


    def extract_3d_scans_per_sub_folder(self, path_sub_folder, shuffle=True, limit=None):
        path_sub_folder = self.check_path(path_sub_folder)
        if limit is None:
            limit = int(1e9)
        paths_scans = list(path_sub_folder.rglob('*.ply'))
        if shuffle:
            random.shuffle(paths_scans)
        generator_scans_paths = (path for path in paths_scans)
        for idx, ply_file in enumerate(generator_scans_paths):
            if idx > limit:
                break
            scan_3d = self.extract_scan(ply_file)
            yield scan_3d


    def extract_scan(self, path_ply):
        face3dscan = None
        with path_ply.open('rb') as reader:
            face3dscan = PlyData.read(reader)

        return face3dscan


class SubDataset300W(Dataset):


    _pattern_version = re.compile('version')
    _pattern = re.compile('n_points')


    def __init__(self, path_dir):
        super().__init__(path_dir)


    def _image_generator_afw(self):
        ext_params = '.pts'
        ext_images = '.jpg'
        generator_img_paths = self.path_dir.rglob(f'*{ext_images}')
        generator_pts_paths = self.path_dir.rglob(f'*{ext_params}')
        image_files = sorted([(int(file.stem.split('_')[0]), file) for file in generator_img_paths])
        pts_files = sorted([(int(file.stem.split('_')[0]), file) for file in generator_pts_paths])
        pts_files = [path for _, path in pts_files]
        image_files = [path for _, path in image_files]

        for idx_img_path, (path_img_file, path_pts_file) in enumerate(zip(image_files, pts_files)):
            landmarks = self._landmarks_file_preprocessor(path_pts_file)
            rgb_image = self._img_file_preprocessor(path_img_file)
            yield (landmarks, rgb_image, path_img_file.stem)


    def _img_file_preprocessor(self, path_img_file, extension='.jpg'):
        img_bgr = cv.imread(str(path_img_file))
        return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


    def _landmarks_file_preprocessor(self, path_pts_file, number_landmarks=68):
        """
        extract 68 pts landmarks from .pts file
        Args:
            path_pts_file:
        Returns:
        np.array (68, 2, dtype=np.float)
        """
        landmarks = np.empty((number_landmarks, 2), dtype=np.float)
        num_pts = 0
        with path_pts_file.open('rb') as reader:
            for idx_line, line in enumerate(reader):
                if idx_line >= 3:
                    try:
                        landmarks[num_pts, :] = [float(coor) for coor in line.decode('utf-8').strip('\n').split(' ')]
                        num_pts += 1
                    except Exception as e:
                        logging.info(f'Incorrect line format: {line} idx line: {idx_line}!')
        return landmarks


class FaceDataset300WLP():


    def __init__(self, directory_data):
        if isinstance(directory_data, str):
            directory_data = Path(directory_data)
        if not directory_data.is_dir():
            raise NotADirectoryError
        self._directory_path = directory_data


    def extract_3d_points(self, subfolder='AFW'):
        subfolder = self._directory_path / subfolder
        if not subfolder.is_dir():
            raise NotADirectoryError(f'Directory: {subfolder} not found!')
        logging.info(f'Check subfolder: {subfolder}.')
        matlab_files = subfolder.rglob('*.mat')
        # sort by order
        files = sorted([(int(file.stem.split('_')[0]), file) for file in matlab_files])
        for idx_sample, (id_file, path_file) in enumerate(files):
            matrix_sample = scipy.io.loadmat(str(path_file))
            matrix_fitted_face = matrix_sample['Fitted_Face']
            yield matrix_fitted_face, path_file.stem
