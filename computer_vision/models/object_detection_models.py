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

    def resize_bboxes(self, resized_img_height, resized_img_width) -> 'ImageData':
        new_bboxes = []
        for bbox_num, bbox in enumerate(self.bboxes):
            # get the GT box coordinates, and resize to account for image resizing
            new_x1 = bbox.x1 * (resized_img_width / float(self.img_width))
            new_x2 = bbox.x2 * (resized_img_width / float(self.img_width))
            new_y1 = bbox.y1 * (resized_img_height / float(self.img_height))
            new_y2 = bbox.y2 * (resized_img_height / float(self.img_height))
            new_bbox = BBox(x1=new_x1, x2=new_x2, y1=new_y1, y2=new_y2, class_name=bbox.class_name)
            new_bboxes.append(new_bbox)

        return ImageData(img_file_name=self.img_file_name, img_file_path=self.img_file_path, img_width=resized_img_width,
                         img_height=resized_img_height, bboxs=new_bboxes)
