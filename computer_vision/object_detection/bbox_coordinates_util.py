def get_bottom_left_coordinate_of_bbox(x1, y1, x2, y2):
    # x1, y1: the top left coordinates
    # x2, y2: the bottom right coordinates

    return x1, y2


def get_width_and_height_of_bbox(x1, y1, x2, y2):
    # x1, y1: the top left coordinates
    # x2, y2: the bottom right coordinates

    return x2 - x1, y2 - y1