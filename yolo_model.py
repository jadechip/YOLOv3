from __future__ import division

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Utils
def list_lookup(x): return x - 1


class Darknet(nn.Module):
    def __init__(self, learning_rate=.1):
        super(Darknet, self).__init__()

        self.learning_rate = learning_rate
        self.history = []

        # Init classifiers
        self.classifier_1 = YOLO(mask=(6, 7, 8),
                                 anchors=[(116, 90),
                                          (156, 198),
                                          (373, 326)],
                                 classes=80,
                                 num=9,
                                 jitter=.3,
                                 ignore_thresh=.7,
                                 truth_thresh=1,
                                 random=1)

        self.classifier_2 = YOLO(mask=(6, 7, 8),
                                 anchors=[
            (30, 61),
            (62, 45),
            (59, 119)],
            classes=80,
            num=9,
            jitter=.3,
            ignore_thresh=.7,
            truth_thresh=1,
            random=1)

        self.classifier_3 = YOLO(mask=(6, 7, 8),
                                 anchors=[(10, 13),
                                          (16, 30),
                                          (33, 23)],
                                 classes=80,
                                 num=9,
                                 jitter=.3,
                                 ignore_thresh=.7,
                                 truth_thresh=1,
                                 random=1)

    def _convolution(self, x, input_size, output_size, kernel_size, stride, padding, bias=False):

        # torch.nn.Conv2d
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        res = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size,
                      stride, padding, bias=bias),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(self.learning_rate),
        )(x)
        self.history.append(res)
        return res

    def _shortcut(self, index):
        return self.history[-1] + self.history[index]

    def _route(self, indices):
        # Returns concatenation of feature maps from given indices
        feature_maps = [self.history[x] for x in indices]
        return torch.cat(feature_maps, 1)

    def _upsample(self, x):
        # (Pdb) x.shape
        # torch.Size([1, 255, 29, 29])
        # (Pdb) y.shape
        # torch.Size([1, 255, 58, 58])
        return nn.Upsample(scale_factor=2, mode="bilinear")(x)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints

        x = self._convolution(x, 3, 32, 3, 1, 1)
        x = self._convolution(x, 32, 64, 3, 2, 1)

        x = self._convolution(x, 64, 32, 1, 1, 0)
        x = self._convolution(x, 32, 64, 3, 1, 1)
        x = self._shortcut(-3)

        x = self._convolution(x, 64, 128, 3, 2, 1)

        for _ in range(2):
            x = self._convolution(x, 128, 64, 1, 1, 0)
            x = self._convolution(x, 64, 128, 3, 1, 1)
            x = self._shortcut(-3)

        x = self._convolution(x, 128, 256, 3, 2, 1)

        for _ in range(8):
            x = self._convolution(x, 256, 128, 1, 1, 0)
            x = self._convolution(x, 128, 256, 3, 1, 1)
            x = self._shortcut(-3)

        x = self._convolution(x, 256, 512, 3, 2, 1)

        for _ in range(8):
            x = self._convolution(x, 512, 256, 1, 1, 0)
            x = self._convolution(x, 256, 512, 3, 1, 1)
            x = self._shortcut(-3)

        x = self._convolution(x, 512, 1024, 3, 2, 1)

        for _ in range(4):
            x = self._convolution(x, 1024, 512, 1, 1, 0)
            x = self._convolution(x, 512, 1024, 3, 1, 1)
            x = self._shortcut(-3)

        # 1st YOLO
        x = self._convolution(x, 1024, 512, 1, 1, 1)
        x = self._convolution(x, 512, 1024, 3, 1, 1)
        x = self._convolution(x, 1024, 512, 1, 1, 1)
        x = self._convolution(x, 512, 1024, 3, 1, 1)
        x = self._convolution(x, 1024, 512, 1, 1, 1)
        x = self._convolution(x, 512, 1024, 3, 1, 1)

        # TODO: Linear activation
        x = self._convolution(x, 1024, 255, 1, 1, 1)
				# self.n_anchors * (self.n_classes + 5) ??
        x = self.classifier_1(x)
        self.history.append(x)

        # Second classification
        x = self._route([-4])
        x = self._convolution(x, x.shape[1], 256, 1, 1, 1)
        x = self._upsample(x)
        x = self._route([-1, list_lookup(61)])
        x = self._convolution(x, x.shape[1], 256, 1, 1, 1)
        x = self._convolution(x, 256, 512, 3, 1, 1)
        x = self._convolution(x, 512, 256, 1, 1, 1)
        x = self._convolution(x, 256, 512, 3, 1, 1)
        x = self._convolution(x, 512, 256, 1, 1, 1)
        x = self._convolution(x, 256, 512, 3, 1, 1)

        # TODO: Linear activation
        x = self._convolution(x, 512, 255, 1, 1, 1)

        x = self.classifier_2(x)
        self.history.append(x)

        # Third classification
        x = self._route([-4])
        x = self._convolution(x, x.shape[1], 255, 1, 1, 1)
        x = self.classifier_3(x)

        return x


class YOLO(nn.Module):
    def __init__(self,
                 mask,
                 anchors,
                 classes,
                 num,
                 jitter,
                 ignore_thresh,
                 truth_thresh,
                 random):
        super(YOLO, self).__init__()

        self.anchors = anchors

    def _route(self, indices):
        # Returns concatenation of feature maps from given indices
        feature_maps = [self.history[x] for x in indices]
        return torch.cat(feature_maps, 1)

    def forward(self, x, targets=None):

        # Dueing training, YOLO layers  should return x, and loss (['xy', 'wh', 'conf', 'cls', 'l2'])


				# Predict loss


				"""
				Adjust prediction
				"""
				### adjust x and y      
				pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

				### adjust w and h
				pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])

				### adjust confidence
				pred_box_conf = tf.sigmoid(y_pred[..., 4])

				### adjust class probabilities
				pred_box_class = y_pred[..., 5:]


#


        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.




				train = targets is not None


        return x
        # route
        # torch.cat([self.layers, new_features], 1)

        # x = self._convolution(x, weights=(3, 32, 3, 1, 1))
        # x = self._route()
        # x = self._convolution(x, weights=(3, 32, 3, 1, 1))


def calculate_iou(xy_min1, xy_max1, xy_min2, xy_max2):
        # "Calculating the IoU is simple we basically divide the overlap area
        #  between the boxes by the union of those areas."

        # Get areas
    areas_1 = np.multiply.reduce(xy_max1 - xy_min1)
    areas_2 = np.multiply.reduce(xy_max2 - xy_min2)

    # determine the (x, y)-coordinates of the intersection rectangle
    _xy_min = np.maximum(xy_min1, xy_min2)
    _xy_max = np.minimum(xy_max1, xy_max2)
    _wh = np.maximum(_xy_max - _xy_min, 0)

    # compute the area of intersection rectangle
    _areas = np.multiply.reduce(_wh)

    # return the intersection over union value
    return _areas / np.maximum(areas_1 + areas_2 - _areas, 1e-10)


def non_max_suppression(conf, xy_min, xy_max, threshold=.4):
    _, _, classes = conf.shape
    # List Comprehension
    # https://www.youtube.com/watch?v=HobjHIpLhZk
    # https://www.youtube.com/watch?v=Q7EYKuZJfdA
    boxes = [(_conf, _xy_min, _xy_max) for _conf, _xy_min, _xy_max in zip(
        conf.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]

    # Iterate each class
    for c in range(classes):
        # Sort boxes
        boxes.sort(key=lambda box: box[0][c], reverse=True)
        # Iterate each box
        for i in range(len(boxes) - 1):
            box = boxes[i]
            if box[0][c] == 0:
                continue
            for _box in boxes[i + 1:]:
                # Take iou threshold into account
                if calculate_iou(box[1], box[2], _box[1], _box[2]) >= threshold:
                    _box[0][c] = 0
    return boxes


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (256, 256))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


#  Tests
test_input = get_test_input()
model = Darknet()
output = model(test_input)

writer.close()
# 1/32
# 1/16
# 1/8
# assert output.shape[2] === 256/32


