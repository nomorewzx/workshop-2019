from typing import List

import cv2
import os

from models.object_detection_models import ImageData, BBox


def get_data(annotation_file_path, img_dir):
    """Parse the data from annotation file
    Args:
        annotation_file_path: annotation file path
    Returns:
        all_img_data: List[ImageData]
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_img_data: List[ImageData] = []

    class_mapping = {}

    with open(annotation_file_path, 'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            bbox = BBox(x1=int(x1), x2=int(x2), y1=int(y1), y2=int(y2), class_name=class_name)

            if filename not in [img_data.img_file_name for img_data in all_img_data]:
                img_file_path = os.path.join(img_dir, filename)
                img = cv2.imread(img_file_path)
                (rows, cols) = img.shape[:2]

                img_data = ImageData(img_file_path=img_file_path, bboxs=[bbox], img_width=cols,
                                     img_height=rows, img_file_name=filename)

                all_img_data.append(img_data)
            else:
                for img_data in all_img_data:
                    if img_data.img_file_name == filename:
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
