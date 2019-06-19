from typing import List

import cv2

from object_detection.models import ImageData, BBox


def get_data(input_path):
    """Parse the data from annotation file
    Args:
        input_path: annotation file path
    Returns:
        all_img_data: List[ImageData]
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_img_data: List[ImageData] = []

    class_mapping = {}

    with open(input_path, 'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            #	One path_filename might has several classes (class_name)
            #	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
            #	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates

            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            bbox = BBox(x1=int(x1), x2=int(x2), y1=int(y1), y2=int(y2), class_name=class_name)

            if filename not in [img_data.img_file_path for img_data in all_img_data]:
                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]

                img_data = ImageData(img_file_path=filename, bboxs=[bbox], img_width=cols, img_height=rows)
                all_img_data.append(img_data)
            else:
                for img_data in all_img_data:
                    if img_data.img_file_path == filename:
                        img_data.add_bbox(bbox)
                        break

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_img_data, class_mapping
