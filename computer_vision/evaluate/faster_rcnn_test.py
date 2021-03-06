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

from object_detection.faster_rcnn.neural_network_components import rpn_to_roi
from object_detection.ground_truth_anchor_generator import get_new_img_size


class FasterRcnnEvaluationTest(unittest.TestCase):
    @unittest.skip('test model')
    def test_evaluate_rpn_of_faster_rcnn(self):
        show_all_rpn_bboxes = True
        valid_box_count = 0
        display_n_th_box = 5

        config = Config()
        config.model_path = os.path.join(settings.MODEL_WEIGHTS_DIR, 'faster_rcnn', 'model_frcnn_vgg.hdf5')

        img_file_path = os.path.join(settings.PROJECT_BASE_DIR, 'object_detection', 'resources', 'c6dad5afbd0ff7a1.jpg')

        to_predict_img = cv2.imread(img_file_path)

        height, width = to_predict_img.shape[:2]

        resized_width, resized_height = get_new_img_size(width=width, height=height)

        test_img = cv2.resize(to_predict_img, (resized_width, resized_height))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        predicted_rpn = evaluate_faster_rcnn(config=config, img=test_img)
        # [(classification_output), (regression_output)]
        rpn_classes = predicted_rpn[0] # [[x,y,anchor_idx],[],[]]
        rpn_regression = predicted_rpn[1]

        fig: Figure = plt.figure()
        axes: Axes = plt.axes()
        axes.imshow(test_img)

        positive_prediction = np.where(rpn_classes > 0.7)

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

            if not self._is_bbox_in_img(x1=anchor_x1, x2=anchor_x2, y1=anchor_y1, y2=anchor_y2,
                                    img_height=resized_height, img_width=resized_height):
                print('this anchor is out of image boundary')
                continue
            else:
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

                predicted_bbox_x2 = predicted_bbox_cx + predicted_bbox_width / 2
                predicted_bbox_y2 = predicted_bbox_cy + predicted_bbox_height / 2

                if not self._is_bbox_in_img(x1=predicted_bbox_x1, y1=predicted_bbox_y1, x2=predicted_bbox_x2, y2=predicted_bbox_y2
                                        , img_width=resized_width, img_height=resized_height):
                    print('calculated bbox is not in image range')
                    continue

                valid_box_count += 1
                if not show_all_rpn_bboxes and valid_box_count != display_n_th_box:
                    continue

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

                if not show_all_rpn_bboxes:
                    break

        fig.add_axes(axes)
        plt.show()

    @unittest.skip('test model')
    def test_apply_rpn_to_roi_and_nms_layer(self):
        config = Config()
        config.model_path = os.path.join(settings.MODEL_WEIGHTS_DIR, 'faster_rcnn', 'model_frcnn_vgg.hdf5')

        img_file_path = os.path.join(settings.PROJECT_BASE_DIR, 'object_detection', 'resources', 'c6dad5afbd0ff7a1.jpg')

        to_predict_img = cv2.imread(img_file_path)

        height, width = to_predict_img.shape[:2]

        resized_width, resized_height = get_new_img_size(width=width, height=height)

        test_img = cv2.resize(to_predict_img, (resized_width, resized_height))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        fig: Figure = plt.figure()
        axes: Axes = plt.axes()
        axes.imshow(test_img)

        predicted_rpn = evaluate_faster_rcnn(config=config, img=test_img)
        # [(classification_output), (regression_output)]
        rpn_classes = predicted_rpn[0]  # [[x,y,anchor_idx],[],[]]
        rpn_regression = predicted_rpn[1]

        proposed_bboxes = rpn_to_roi(rpn_cls_layer=rpn_classes, regr_layer=rpn_regression, C=config)

        for bbox in proposed_bboxes:
            x1, y1, x2, y2 = bbox

            x1 = x1 * config.rpn_stride
            x2 = x2 * config.rpn_stride

            y1 = y1 * config.rpn_stride
            y2 = y2 * config.rpn_stride

            bbox_rect = Rectangle((x1, y1),
                                  width=int(x2 - x1),
                                  height=int(y2 - y1),
                                  linewidth=2,
                                  edgecolor='r',
                                  linestyle='--',
                                  facecolor='none')
            axes.add_patch(bbox_rect)

        fig.add_axes(axes)
        plt.show()


    def _is_bbox_in_img(self, x1, y1, x2, y2, img_width, img_height):
        if x1 < 0 or x2 > img_width or y1 < 0 or y2 > img_height:
            return False

        return True