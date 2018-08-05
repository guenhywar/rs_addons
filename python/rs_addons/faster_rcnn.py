import numpy as np
import os.path as osp
import warnings

from chainer.backends import cuda
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
import rospkg


class FasterRCNNDetectionPredictor(object):

    def __init__(
            self, model='faster_rcnn_vgg16',
            pretrained_model='voc0712', gpu=-1, score_thresh=0.3
    ):
        if model == 'faster_rcnn_vgg16':
            model_class = FasterRCNNVGG16
        else:
            warnings.warn('no model class: {}'.format(model))

        r = rospkg.RosPack()
        if pretrained_model == 'voc07' and model == 'faster_rcnn_vgg16':
            self.label_names = voc_bbox_label_names
            pretrained_model = osp.join(
                r.get_path('rs_addons'),
                'trained_data/faster_rcnn_vgg16_voc07_trained.npz')
        elif pretrained_model == 'voc0712' and model == 'faster_rcnn_vgg16':
            self.label_names = voc_bbox_label_names
            pretrained_model = osp.join(
                r.get_path('rs_addons'),
                'trained_data/faster_rcnn_vgg16_voc0712_trained.npz')
        else:
            warnings.warn('no pretrained model: {}'.format(pretrained_model))

        self.model = model_class(
            n_fg_class=len(self.label_names),
            pretrained_model=pretrained_model)
        self.model.score_thresh = score_thresh
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def predict(self, img):
        img = img[:, :, ::-1].transpose((2, 0, 1))
        imgs = img[None]
        bboxes, labels, scores = self.model.predict(imgs)
        bbox, label, score = bboxes[0], labels[0], scores[0]
        bbox = np.round(bbox).astype(np.int32)
        return bbox, label, score
