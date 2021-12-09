import dlib
import logging
import numpy as np
import os
from pathlib import Path


class LoaderDlibFaceModels():

    names_face_recognition_models = ['dlib_face_recognition_resnet_model_v1.dat']


    def __init__(self, directory_models):
        if isinstance(directory_models, str) and os.path.isdir(directory_models):
            directory_models = Path(directory_models)
        self.directory_models = directory_models


    def get_landmarks_detector_dlib(self, number_landmark_points=5):
        if number_landmark_points not in [5, 68]:
            assert False, f'Models with 68 and 5 key points are available. But was reciewed: {number_landmark_points}.'
        filepath = self.directory_models / f'shape_predictor_{number_landmark_points}_face_landmarks.dat'
        if not filepath.is_file():
            assert False, f'The key points prediction model with {number_landmark_points} landmarks was not found!'
        return dlib.shape_predictor(str(filepath))


    def get_face_recog_model_dlib(self):
        filepath = self.directory_models / 'dlib_face_recognition_resnet_model_v1.dat'
        if not filepath.is_file():
            assert False, f'The facial recognition model was not found! The file with the path: {filepath}.'
        return dlib.face_recognition_model_v1(str(filepath))



class FaceRecognizerDlib():


    def __init__(self, directory_models, landmark_model='big'):
        self.directory_models = directory_models
        self.landmark_model = landmark_model
        self.shape_model, self.face_recognition_net = self.load_dlib_models()


    def load_dlib_models(self):
        loader_face_recog_models = LoaderDlibFaceModels(self.directory_models)
        number_landmarks = 5 if self.landmark_model == 'small' else 68
        logging.info(f'We load {self.landmark_model} model with {number_landmarks} landmarks method extracter!')
        shape_model = loader_face_recog_models.get_landmarks_detector_dlib(number_landmarks)
        face_recognition_net = loader_face_recog_models.get_face_recog_model_dlib()
        return shape_model, face_recognition_net


    def get_face_embeddings(self, frame_rgb, boxes, return_aligned_faces=True):
        embeddings = []
        aligned_faces = []
        for face_idx, box in enumerate(boxes):
            #start = time.time()
            embedding, aligned_face = self.get_face_embedding(frame_rgb, box, return_aligned_faces)
            embedding = np.asarray(embedding)
            #print(f'Time dlib face_embeddings: {time.time() - start:.4f} s')
            embeddings.append(embedding)
            aligned_faces.append(aligned_face)
        return embeddings, aligned_faces


    def get_face_embedding(self, frame_rgb, box, return_aligned_faces=True):
        face_landmarks = self.get_face_landmarks(frame_rgb, box)
        aligned_face = self.get_aligned_face(frame_rgb, face_landmarks)
        face_embedding = self.face_recognition_net.compute_face_descriptor(aligned_face)
        if return_aligned_faces:
            return face_embedding, aligned_face
        else:
            return face_embedding


    def get_aligned_face(self, frame_rgb, shape_face_landmarks):
        return dlib.get_face_chip(frame_rgb, shape_face_landmarks)


    def get_face_landmarks(self, frame_rgb, box):
        x1, y1, x2, y2 = box
        box_dlib = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        return self.shape_model(frame_rgb, box_dlib)
