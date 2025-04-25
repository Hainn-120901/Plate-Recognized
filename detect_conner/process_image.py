import numpy as np
import cv2


def get_center_point(coordinate_dict):
    """ Tính toán trung điểm của các bounding box """
    return {
        key: ((xmin + xmax) / 2, (ymin + ymax) / 2)
        for key, (xmin, ymin, xmax, ymax, *_) in coordinate_dict.items()
    }


def find_miss_corner(coordinate_dict):
    " Xác định góc bị thiếu "
    full_set = {0, 1, 2, 3}
    detected_set = set(coordinate_dict.keys())
    return list(full_set - detected_set)


def calculate_missed_coord_corner(coordinate_dict):
    """ Tính toán vị trí góc bị thiếu bằng nội suy """
    missing = find_miss_corner(coordinate_dict)
    if not missing:
        return coordinate_dict

    known_corners = sorted(list(coordinate_dict.keys()))
    coords = np.array([coordinate_dict[k] for k in known_corners])
    for miss in missing:
        if miss == 0:  # top_left
            coordinate_dict[0] = (coords[2][0] - coords[0][0] + coords[1][0], coords[2][1] - coords[1][1] + coords[0][1])
        elif miss == 1:  # top_right
            coordinate_dict[1] = (coords[1][0] + coords[2][0] - coords[0][0], coords[1][1] - coords[2][1] + coords[0][1])
        elif miss == 2:  # bottom_left
            coordinate_dict[2] = (coords[1][0] - coords[2][0] + coords[0][0], coords[1][1] + coords[2][1] - coords[0][1])
        elif miss == 3:  # bottom_right
            coordinate_dict[3] = (coords[0][0] + coords[1][0] - coords[2][0], coords[0][1] + coords[2][1] - coords[1][1])

    return coordinate_dict


def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    return cv2.warpPerspective(image, M, (500, 300))


def add_pad(image, pad_ratio=0.1):
    """ Thêm padding vào ảnh """
    old_h, old_w, channels = image.shape
    new_w = int(old_w * (1 + pad_ratio))
    new_h = int(old_h * (1 + pad_ratio))

    result = np.zeros((new_h, new_w, channels), dtype=np.uint8)
    x_offset = (new_w - old_w) // 2
    y_offset = (new_h - old_h) // 2

    result[y_offset:y_offset + old_h, x_offset:x_offset + old_w] = image
    return result


def align_image(image, coordinate_dict):
    """ Căn chỉnh ảnh dựa vào góc phát hiện """
    coordinate_dict = get_center_point(coordinate_dict)
    if len(coordinate_dict) <= 2:
        return None  # Không đủ góc để căn chỉnh
    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)
    # Chuyển đổi danh sách tọa độ theo thứ tự chuẩn
    source_points = np.float32([
        coordinate_dict[0],  # top_left
        coordinate_dict[1],  # top_right
        coordinate_dict[3],  # bottom_right
        coordinate_dict[2]   # bottom_left
    ])

    return perspective_transform(image, source_points)
