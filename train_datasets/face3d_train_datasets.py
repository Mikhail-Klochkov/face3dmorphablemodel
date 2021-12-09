import logging
import scipy.io
from pathlib import Path
import re
import cv2 as cv
import numpy as np



class Dataset():


    def __init__(self, path_dir):
        self._path_dir = self._check_directory_path(path_dir)


    def _check_directory_path(self, path_dir):
        if isinstance(path_dir, str):
            path_dir = Path(path_dir)
        if not path_dir.is_dir():
            raise NotADirectoryError
        return path_dir



class SubDataset300W(Dataset):


    _pattern_version = re.compile('version')
    _pattern = re.compile('n_points')


    def __init__(self, path_dir):
        super().__init__(path_dir)


    def _image_generator_afw(self):
        ext_params = '.pts'
        ext_images = '.jpg'
        generator_img_paths = self._path_dir.rglob(f'*{ext_images}')
        generator_pts_paths = self._path_dir.rglob(f'*{ext_params}')
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
