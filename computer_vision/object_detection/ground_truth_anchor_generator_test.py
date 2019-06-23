import unittest

from models.object_detection_models import BBox, ImageData
from object_detection.config import Config
from object_detection.ground_truth_anchor_generator import calc_rpn, get_new_img_size
from object_detection.img_debug_display import display_img_data


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

        rpn_class, rpn_regression, num_pos = calc_rpn(C=config, img_data=img_data, width=width, height=height,
                                                      resized_width=resized_width, resized_height=resized_height,
                                                      img_length_calc_function=get_new_img_size)

        print(rpn_class)
        print(rpn_regression)
        print(num_pos)

    def _get_img_data(self):
        raw_bbox_1 = BBox(x1=571, y1=474, x2=667, y2=743, class_name='Person')
        raw_bbox_2 = BBox(x1=39, y1=476, x2=400, y2=578, class_name='Car')

        raw_bboxes = [raw_bbox_1, raw_bbox_2]

        img_data = ImageData(img_file_name='c6dad5afbd0ff7a1.jpg',
                             img_file_path='resources/c6dad5afbd0ff7a1.jpg',
                             bboxs=raw_bboxes, img_height=765, img_width=1024)

        return img_data
