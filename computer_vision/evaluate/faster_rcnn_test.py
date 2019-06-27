import unittest
import os

from evaluate.faster_rcnn import evaluate_faster_rcnn
from object_detection.config import Config
import settings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from object_detection.ground_truth_anchor_generator import get_new_img_size


class FasterRcnnEvaluationTest(unittest.TestCase):
    def test_evaluate_faster_rcnn(self):
        config = Config()
        config.model_path = os.path.join(settings.MODEL_WEIGHTS_DIR, 'faster_rcnn', 'model_frcnn_vgg.hdf5')

        img_file_path = os.path.join(settings.PROJECT_BASE_DIR, 'object_detection', 'resources', '0e9b73b26eb837ec.jpg')

        to_predict_img = cv2.imread(img_file_path)

        height, width = to_predict_img.shape[:2]

        resized_width, resized_height = get_new_img_size(width=width, height=height)

        test_img = cv2.resize(to_predict_img, (resized_width, resized_height))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        predicted_rpn = evaluate_faster_rcnn(config=config, img=test_img)
        rpn_classes = predicted_rpn[0]
        rpn_regression = predicted_rpn[1]

        fig: Figure = plt.figure()
        axes: Axes = plt.axes()
        axes.imshow(test_img)

        positive_prediction = np.where(rpn_classes > 0.99)
        print(positive_prediction[-1])
        print(len(positive_prediction[0]))
        valid_box_count = 0
        display_n_th_box = 20
        for i in range(len(positive_prediction[0])):
            img_idx = positive_prediction[0][i]
            row_idx = positive_prediction[1][i]
            col_idx = positive_prediction[2][i]
            anchor_idx = positive_prediction[3][i]
            anchor_box_scal_idx = anchor_idx // len(config.anchor_box_scales)
            anchor_box_ratio_idx = anchor_idx % len(config.anchor_box_scales)

            anchor_x = config.anchor_box_scales[anchor_box_scal_idx] * config.anchor_box_ratios[anchor_box_ratio_idx][0]
            anchor_y = config.anchor_box_scales[anchor_box_scal_idx] * config.anchor_box_ratios[anchor_box_ratio_idx][1]

            anchor_x1 = config.rpn_stride * (col_idx + 0.5) - anchor_x / 2
            anchor_x2 = config.rpn_stride * (col_idx + 0.5) + anchor_x / 2

            anchor_y1 = config.rpn_stride * (row_idx + 0.5) - anchor_y / 2
            anchor_y2 = config.rpn_stride * (row_idx + 0.5) + anchor_y / 2

            if anchor_x1 < 0 or anchor_x2 > resized_width or anchor_y1 < 0 or anchor_y2 > resized_height:
                print('this anchor is out of image boundary')
                continue
            else:
                valid_box_count += 1
                if valid_box_count != display_n_th_box:
                    continue
                print('find valid positive anchor at row: {}, col: {}, anchor_idx: {}'.format(row_idx, col_idx, anchor_idx))
                anchor_cx = (anchor_x1 + anchor_x2) / 2.0
                anchor_cy = (anchor_y1 + anchor_y2) / 2.0

                tx, ty, tw, th = rpn_regression[img_idx][row_idx][col_idx][4 * anchor_idx: 4 * anchor_idx + 4]
                predicted_bbox_cx = tx * anchor_x + anchor_cx
                predicted_bbox_cy = ty * anchor_y + anchor_cy

                predicted_bbox_width = np.exp(tw) * anchor_x
                predicted_bbox_height = np.exp(ty) * anchor_y

                predicted_bbox_x1 = predicted_bbox_cx - predicted_bbox_width / 2
                predicted_bbox_y1 = predicted_bbox_cy - predicted_bbox_height / 2

                print('the regression is {}'.format((tx, ty, tw, th)))
                print('anchor x1: {}, y1: {}'.format(anchor_x1, anchor_y1))
                print('pred_bbox x1: {}, y1: {}'.format(predicted_bbox_x1, predicted_bbox_y1))

                bbox_rect = Rectangle((predicted_bbox_x1, predicted_bbox_y1),
                                      width=int(predicted_bbox_width),
                                      height=int(predicted_bbox_height),
                                      linewidth=2,
                                      edgecolor='r',
                                      linestyle='--',
                                      facecolor='none')
                axes.add_patch(bbox_rect)
                break
        fig.add_axes(axes)
        plt.show()