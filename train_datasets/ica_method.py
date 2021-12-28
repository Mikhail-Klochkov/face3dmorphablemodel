import time, numpy as np, logging

from pathlib import Path
from face3d_train_datasets import FaceDataset300WLP
from sklearn.decomposition import FastICA


# not doing it
class ICA300WL():


    def __init__(self, dictionary_data, random_state=20):
        self.path_dictionary = Path(dictionary_data)
        self.facedataset = FaceDataset300WLP(self.path_dictionary)
        self.random_state = random_state


    def fit_ICA(self, number_components=200, limit=None):
        X = self.extract_data_matrix_shapes(limit=limit)
        fastica = FastICA(n_components=number_components, random_state=self.random_state)
        start = time.time()
        X_transformed = fastica.fit_transform(X)
        logging.info(f'Time fitting: {time.time()-start} s' )


    def extract_data_matrix_shapes(self, limit=None):
        if limit is None:
            limit = int(1e9)
        X = []
        for idx, (points_3d, name_file) in enumerate(self.facedataset.extract_3d_points(), 1):
            if idx > limit:
                break
            points_3d = points_3d.T
            X.append(points_3d)

        X = np.asarray(X)

        return X


    def check_path(self, path):
        if isinstance(path, str):
            path = Path(path)
        assert path.is_file() or path.is_dir()
        return path
