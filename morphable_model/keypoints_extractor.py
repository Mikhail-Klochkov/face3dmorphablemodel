from insightface.app import FaceAnalysis



class KeyPointsExtractor(FaceAnalysis):


    def __init__(self, ctx_id=0, det_size = (640, 640)):
        super().__init__()
        self.prepare(ctx_id, det_size)


    def get_keypoints(self, img, get_boxes=False):
        try:
            self.det_model = self.__getattribute__('det_model')
        except AttributeError:
            assert 'Class FaceRecognizerWrapper has no attribute det_model!'
        _, keypoints = self.det_model.detect(img)
        return keypoints