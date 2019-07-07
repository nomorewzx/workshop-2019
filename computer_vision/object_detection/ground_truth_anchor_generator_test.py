import unittest

from data_models.object_detection_models import BBox, ImageData
from object_detection.config import Config
from object_detection.ground_truth_anchor_generator import calc_rpn, get_new_img_size, get_img_output_length
import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from settings import MATPLOTLIB_COLORS


class GroundTruthRpnGeneratorTest(unittest.TestCase):
    def test_should_generate_rpn_ground_truth(self):
        config = Config()
        config.rpn_max_overlap = 0.5
        config.rpn_min_overlap = 0.2
        config.class_mapping = {'Person': 0, 'Car': 1, 'Mobile Phone': 2}
        img_data = self._get_img_data()

        width = img_data.img_width
        height = img_data.img_height

        resized_width, resized_height = get_new_img_size(width=width, height=height)

        resized_img_data = img_data.resize_bboxes(resized_img_height=resized_height, resized_img_width=resized_width)

        rpn_class, rpn_regression, num_pos, debug_rpn_anchor_bbxes = calc_rpn(C=config, img_data=img_data,
                                                                              resized_width=resized_width,
                                                                              resized_height=resized_height,
                                                                              img_length_calc_function=get_img_output_length)

        test_img = cv2.imread(resized_img_data.img_file_path)

        read_img_height, read_img_width = test_img.shape[:2]

        if read_img_height != resized_img_data.img_height or read_img_width != resized_img_data.img_width:
            print('need to resize img')
            test_img = cv2.resize(test_img, (resized_img_data.img_width, resized_img_data.img_height))

        # OpenCV read, show, write image in BGR order, to display using matplotlib, need to convert to RGB order
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        fig: Figure = plt.figure()
        axes: Axes = plt.axes()
        axes.imshow(test_img)

        for bbox in resized_img_data.bboxes:
            box_width = bbox.get_width()
            box_height = bbox.get_height()

            bbox_rect = Rectangle((bbox.x1, bbox.y1),
                                  width=box_width, height=box_height,
                                  linewidth=2,
                                  edgecolor=MATPLOTLIB_COLORS[config.class_mapping[bbox.class_name]],
                                  facecolor='none')

            axes.add_patch(bbox_rect)

        generated_anchor_count = 0

        for i in range(rpn_class.shape[0]):
            for m in range(rpn_class.shape[1]):
                for n in range(rpn_class.shape[2]):
                    rpn_class_m_n = rpn_class[i][m][n]
                    if any(rpn_class_m_n):
                        has_positive_anchor = any(rpn_class_m_n[:9] * rpn_class_m_n[9:])
                        if has_positive_anchor:
                            is_valid_anchor = rpn_class_m_n[:9]
                            for positive_anchor_idx in np.where(is_valid_anchor == 1)[0]:
                                generated_anchor_count += 1

                                debug_anchor_bbox_start_idx = positive_anchor_idx * 4
                                anchor_bbox = debug_rpn_anchor_bbxes[i][m][n][
                                              debug_anchor_bbox_start_idx: debug_anchor_bbox_start_idx + 4]

                                bbox_rect = Rectangle((anchor_bbox[0], anchor_bbox[1]),
                                                      width=int(anchor_bbox[2] - anchor_bbox[0]),
                                                      height=int(anchor_bbox[3] - anchor_bbox[1]),
                                                      linewidth=2,
                                                      edgecolor=MATPLOTLIB_COLORS[
                                                          generated_anchor_count % len(MATPLOTLIB_COLORS)],
                                                      linestyle='--',
                                                      facecolor='none')

                                axes.add_patch(bbox_rect)

        fig.add_axes(axes)
        plt.show()

    def _get_img_data(self):
        raw_bbox_1 = BBox(x1=571, y1=474, x2=667, y2=743, class_name='Person')
        raw_bbox_2 = BBox(x1=39, y1=476, x2=400, y2=578, class_name='Car')

        raw_bboxes = [raw_bbox_1, raw_bbox_2]

        img_data = ImageData(img_file_name='c6dad5afbd0ff7a1.jpg',
                             img_file_path='resources/c6dad5afbd0ff7a1.jpg',
                             bboxs=raw_bboxes, img_height=765, img_width=1024)

        return img_data
