import os
import unittest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from object_detection.data_loader import get_data
from models.object_detection_models import ImageData
from settings import MATPLOTLIB_COLORS


class DataLoaderTest(unittest.TestCase):
    def test_load_img_and_annotations(self):


        annotation_file_path = os.path.join(os.path.dirname(__file__), 'resources', 'annotation.txt')
        all_img_data, class_mapping = get_data(annotation_file_path, img_dir= 'resources/')

        test_img_data: ImageData = all_img_data[0]
        test_img = plt.imread(test_img_data.img_file_path)
        fig: Figure = plt.figure()

        axes: Axes = plt.axes()
        axes.imshow(test_img)

        for bbox in test_img_data.bboxes:
            box_width = bbox.get_width()
            box_height = bbox.get_height()

            bbox_rect = Rectangle((bbox.x1, bbox.y1),
                                  width=box_width, height=box_height,
                                  linewidth=2,
                                  edgecolor=MATPLOTLIB_COLORS[class_mapping[bbox.class_name]],
                                  facecolor='none')

            axes.add_patch(bbox_rect)

        fig.add_axes(axes)
        plt.show()
