import unittest

from models.object_detection_models import ImageData, BBox
from object_detection.img_debug_display import display_img_data


class ImageDataTest(unittest.TestCase):
    def test_should_resize_image_data_bboxes(self):
        raw_bbox_1 = BBox(x1=571, y1=474, x2=667, y2=743, class_name='Person')
        raw_bbox_2 = BBox(x1=39, y1=476, x2=400, y2=578, class_name='Car')

        raw_bboxes = [raw_bbox_1, raw_bbox_2]

        img_data = ImageData(img_file_name='c6dad5afbd0ff7a1.jpg',
                             img_file_path='../object_detection/resources/c6dad5afbd0ff7a1.jpg',
                             bboxs=raw_bboxes, img_height=765, img_width=1024)

        resized_img_data = img_data.resize_bboxes(resized_img_width=401, resized_img_height=300)
        display_img_data(resized_img_data)
