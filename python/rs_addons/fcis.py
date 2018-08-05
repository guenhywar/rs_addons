import numpy as np
import os.path as osp
import warnings

from chainer.backends import cuda
from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.experimental.links import FCISResNet101
from chainercv.utils import mask_to_bbox
import rospkg

from rs_addons.coco_utils import coco_instance_segmentation_label_names


def mask_to_roi_mask(mask, bbox):
    roi_mask = []
    for msk, bb in zip(mask, bbox):
        roi_msk = msk[bb[0]:bb[2], bb[1]:bb[3]]
        roi_mask.append(roi_msk)
    return roi_mask


class FCISInstanceSegmentationPredictor(object):

    def __init__(
            self, model='fcis_resnet101', pretrained_model='sbd',
            gpu=-1, score_thresh=0.3):
        if model == 'fcis_resnet101':
            model_class = FCISResNet101
        else:
            warnings.warn('no model class: {}'.format(model))

        r = rospkg.RosPack()
        if pretrained_model == 'sbd' and model == 'fcis_resnet101':
            self.label_names = sbd_instance_segmentation_label_names
            pretrained_model = osp.join(
                r.get_path('rs_addons'),
                'trained_data/fcis_resnet101_sbd_trained.npz')
            self.model = model_class(
                n_fg_class=len(self.label_names),
                pretrained_model=pretrained_model)
        elif pretrained_model == 'coco' and model == 'fcis_resnet101':
            self.label_names = coco_instance_segmentation_label_names
            pretrained_model = osp.join(
                r.get_path('rs_addons'),
                'trained_data/fcis_resnet101_coco_trained.npz')
            proposal_creator_params = {
                'nms_thresh': 0.7,
                'n_train_pre_nms': 6000,
                'n_train_post_nms': 300,
                'n_test_pre_nms': 6000,
                'n_test_post_nms': 300,
                'force_cpu_nms': False,
                'min_size': 2
            }
            self.model = model_class(
                n_fg_class=len(self.label_names),
                anchor_scales=(4, 8, 16, 32),
                proposal_creator_params=proposal_creator_params,
                pretrained_model=pretrained_model)
        else:
            warnings.warn('no pretrained model: {}'.format(pretrained_model))

        self.model.score_thresh = score_thresh
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def predict(self, img):
        img = img[:, :, ::-1].transpose((2, 0, 1))
        imgs = img[None]
        masks, labels, scores = self.model.predict(imgs)
        mask, label, score = masks[0], labels[0], scores[0]
        bbox = mask_to_bbox(mask)
        bbox = np.round(bbox).astype(np.int32)
        mask = (mask * 255).astype(np.uint8)
        roi_mask = mask_to_roi_mask(mask, bbox)
        return roi_mask, bbox, label, score
