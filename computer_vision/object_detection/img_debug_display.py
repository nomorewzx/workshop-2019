import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from models.object_detection_models import ImageData
from settings import MATPLOTLIB_COLORS
import cv2


def display_img_data(img_data: ImageData, class_mapping=None):
    if class_mapping is None:
        class_mapping = {'Person': 0, 'Car': 1}

    test_img = cv2.imread(img_data.img_file_path)

    read_img_height, read_img_width = test_img.shape[:2]

    if read_img_height != img_data.img_height or read_img_width != img_data.img_width:
        print('need to resize image')
        test_img = cv2.resize(test_img, (img_data.img_width, img_data.img_height))

    # OpenCV read, show, write image in BGR order, to display using matplotlib, need to convert to RGB order
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    fig: Figure = plt.figure()
    axes: Axes = plt.axes()
    axes.imshow(test_img)

    for bbox in img_data.bboxes:
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