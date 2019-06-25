import os
import unittest

from data_models.object_detection_models import ImageData
from object_detection.data_loader import get_data
from object_detection.img_debug_display import display_img_data
import settings


class DataLoaderTest(unittest.TestCase):
    def test_load_img_and_annotations(self):
        annotation_file_path = os.path.join(settings.PROJECT_BASE_DIR, 'object_detection', 'resources', 'annotation.txt')
        img_dir = os.path.join(settings.PROJECT_BASE_DIR, 'object_detection', 'resources')
        all_img_data, class_mapping = get_data(annotation_file_path, img_dir=img_dir)

        test_img_data: ImageData = all_img_data[0]

        display_img_data(test_img_data, class_mapping=class_mapping)
