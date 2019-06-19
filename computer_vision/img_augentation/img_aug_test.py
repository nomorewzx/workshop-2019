import unittest

from img_augentation.img_aug import augment
from object_detection.config import Config
from object_detection.data_loader import get_data
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from settings import MATPLOTLIB_COLORS


class ImgAugTest(unittest.TestCase):
    def test_img_aug(self):
        all_img_data, class_mapping = get_data('../object_detection/resources/annotation.txt', '../object_detection/resources')

        config = Config()
        config.use_horizontal_flips = True
        raw_img_data = all_img_data[0]
        aug_img_data, aug_img = augment(raw_img_data, config, augment=True)

        # convert BGR --> RGB
        aug_img = aug_img[:, :, (2, 1, 0)]

        figure = plt.figure()
        axes_raw_img = figure.add_subplot(121)
        axes_aug_img = figure.add_subplot(122)

        raw_img = plt.imread(raw_img_data.img_file_path)
        axes_raw_img.imshow(raw_img)

        for bbox in raw_img_data.bboxes:
            bbox_rect = Rectangle((bbox.x1, bbox.y1), width=bbox.get_width(), height=bbox.get_height(), linewidth=2,
                                  edgecolor=MATPLOTLIB_COLORS[class_mapping[bbox.class_name]], facecolor='none')
            axes_raw_img.add_patch(bbox_rect)

        axes_aug_img.imshow(aug_img)
        for bbox in aug_img_data.bboxes:
            bbox_rect = Rectangle((bbox.x1, bbox.y1), width=bbox.get_width(), height=bbox.get_height(), linewidth=2,
                                  edgecolor=MATPLOTLIB_COLORS[class_mapping[bbox.class_name]], facecolor='none')
            axes_aug_img.add_patch(bbox_rect)

        figure.add_subplot()
        figure.add_axes(axes_raw_img)
        figure.add_axes(axes_aug_img)
        plt.show()