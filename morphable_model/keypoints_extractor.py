from insightface.app import FaceAnalysis



class KeyPointsExtractor(FaceAnalysis):


    def __init__(self, ctx_id=0, det_size = (640, 640)):
        super().__init__()
        self.prepare(ctx_id, det_size)


    def get_keypoints(self, img):
        try:
            self.det_model = self.__getattribute__('det_model')
        except AttributeError:
            assert 'Class FaceRecognizerWrapper has no attribute det_model!'
        _, keypoints = self.det_model.detect(img)
        return keypoints


class FaceDetectorWrapper(FaceAnalysis):


    def __init__(self, threshold = 0.6):
        super().__init__()
        self.det_thresh = threshold


    def detect_faces_wrapper(self, img, max_num = 0, metric ='default'):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric=metric)
        # get score, box, keypoints
        return bboxes[:, -1], bboxes[:, :-1], kpss