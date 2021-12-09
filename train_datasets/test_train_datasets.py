import open3d
from pathlib import Path


from face3d_train_datasets import SubDataset300W, FaceDataset300WLP
from face3d_train_datasets import Dataset
from face3d.my_folder.morphable_model.morphable_model_full_head import MorphableModelFullHead

path_300W_LP = Path('/home/mklochkov/projects/data/3dface/300W-3D-Face/')
path_afw_dataset = Path('/home/mklochkov/projects/data/3dface/afw/')
FULL_HEAD_MODEL_FILE_NAME_2019 = Path('/home/mklochkov/projects/data/3dface/model2019_fullHead.h5')
BASEL_FACE_MODEL_FILE_2019 = Path('/home/mklochkov/projects/data/3dface/model2019_bfm.h5')


class TestTrainDatasets():


    @staticmethod
    def iterate_over_afw(path_afw_dataset=path_afw_dataset):
        afwdataset = SubDataset300W(path_afw_dataset)
        for _ in afwdataset._image_generator_afw():
            pass


    @staticmethod
    def visualize_3d_points_dataset(limit=1):
        model = MorphableModelFullHead(BASEL_FACE_MODEL_FILE_2019)
        facedataset = FaceDataset300WLP(path_300W_LP)
        subdataset_afw = SubDataset300W(path_afw_dataset)
        matrix_3d_points = {}
        for idx, (points_3d, name_file) in enumerate(facedataset.extract_3d_points(), 1):
            matrix_3d_points[name_file] = points_3d
            if idx > limit:
                break
        images = {}
        for idx, (landmark, image, name_file) in enumerate(subdataset_afw._image_generator_afw(), 1):
            images[name_file] = image
            if idx > limit:
                break
        assert list(images.keys()) == list(matrix_3d_points.keys())
        for name_file, img in images.items():
            assert name_file in matrix_3d_points
            points_3d =  matrix_3d_points[name_file]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points_3d.reshape((-1, 3)))
            open3d.visualization.draw_geometries([pcd], width=800, height=1000)



if __name__ == '__main__':
    TestTrainDatasets.visualize_3d_points_dataset()