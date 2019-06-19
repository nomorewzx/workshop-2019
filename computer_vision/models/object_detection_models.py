from typing import List


class BBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, class_name: str):
        # x1,y1, x2, y2 coordinates like below:
        # x1,y1---------------------------
        # |                              |
        # |                              |
        # ----------------------------x2,y2
        self.x1: float = x1
        self.x2: float = x2
        self.y1: float = y1
        self.y2: float = y2
        self.class_name = class_name

    def get_width(self):
        return self.x2 - self.x1

    def get_height(self):
        return self.y2 - self.y1


class ImageData:
    def __init__(self, img_file_path: str, img_file_name: str, bboxs: List[BBox], img_width: int, img_height: int):
        self.img_file_path: str = img_file_path
        self.img_file_name = img_file_name
        self.bboxes: List[BBox] = bboxs
        self.img_width = img_width
        self.img_height = img_height

    def add_bbox(self, bbox: BBox):
        self.bboxes.append(bbox)
