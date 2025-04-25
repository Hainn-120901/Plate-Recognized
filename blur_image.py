import uuid

from PIL import Image, ImageFilter
import numpy as np
import glob
import os

import cv2
import numpy as np


def blur_image(image_path, blur_type='motion', blur_strength=10):

    # Đọc ảnh
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Không thể tải ảnh. Kiểm tra đường dẫn.")

    if blur_type == 'gaussian':
        # Làm mờ Gaussian
        ksize = max(5, blur_strength // 2 * 2 + 1)  # Kích thước kernel phải là số lẻ
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    elif blur_type == 'motion':
        # Tạo kernel giả lập hiệu ứng chuyển động
        kernel = np.zeros((blur_strength, blur_strength))
        kernel[int((blur_strength - 1) / 2), :] = np.ones(blur_strength)
        kernel /= blur_strength

        blurred = cv2.filter2D(img, -1, kernel)

    else:
        raise ValueError("Loại làm mờ không hợp lệ! Chọn 'gaussian' hoặc 'motion'.")

    return blurred

# Đọc ảnh từ file
path_folder = "/home/teamai/Documents/data_test_mo_ro_87k/ro_87k"
path_folder_blur = "/home/teamai/Documents/data_test_mo_ro_87k/data_gen_mo_87k"

l_path_file = glob.glob(os.path.join(path_folder, "*.jpg"))
count = 0
for idx_path, path_image in enumerate(l_path_file):
    blurred_img = blur_image(path_image, blur_type='motion', blur_strength=30)

    cv2.imwrite(os.path.join(path_folder_blur, uuid.uuid4().hex + ".jpg"), blurred_img)
    count += 1
    if count == 1500:
        break



# # Ví dụ sử dụng:
# image_path = "/home/teamai/Documents/data/data_deepface/data_check_face_matching/data_original/data_1k_dau/test_image_blur/image_normal/0b6d521deb1246a58ad7c336f746a8a1_309_.jpg"
# blurred_img = blur_image(image_path, blur_type='gaussian', blur_strength=20)
#
# # # Lưu ảnh kết quả
# # cv2.imwrite("/mnt/data/blurred_image.jpg", blurred_img)
# cv2.imshow("test", blurred_img)
# cv2.waitKey(0)

