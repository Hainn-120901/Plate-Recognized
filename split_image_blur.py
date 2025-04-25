import glob
import os.path
import shutil

import cv2

# Dinh nghia blur threshold
blur_threshold = 200
path_folder = "/home/teamai/Documents/face-ahuy-copy/save_face_matching_13_2_23"
path_folder_blur = "/home/teamai/Documents/data/data_deepface/data_check_face_matching/data_original/data_1k_dau/test_image_blur/split_image/blur"
path_folder_normal = "/home/teamai/Documents/data/data_deepface/data_check_face_matching/data_original/data_1k_dau/test_image_blur/split_image/normal"

l_path_file = glob.glob(os.path.join(path_folder, "*.jpg"))
count = 0
for idx_path, path_image in enumerate(l_path_file):
    print(idx_path, path_image)
    base_name = os.path.basename(path_image)
    # Doc anh tu file
    image = cv2.imread(path_image)
    gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Tinh toan muc do focus cua anh
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

    name_save = base_name.split(".jpg")[0] + "_" + str(int(focus_measure)) + "_.jpg"
    if focus_measure < blur_threshold:
        shutil.copy(path_image, path_folder_blur)
        os.rename(os.path.join(path_folder_blur, base_name), os.path.join(path_folder_blur, name_save))
    else:
        shutil.copy(path_image, path_folder_normal)
        os.rename(os.path.join(path_folder_normal, base_name), os.path.join(path_folder_normal, name_save))

    count += 1
    if count == 5000:
        break

# import cv2
#
# # Duong dan den file anh
# image_file= "/home/teamai/Downloads/9837a62d-7892-43ac-8842-3c72d69187bb.jpeg"
# # Dinh nghia blur threshold
# blur_threshold=100
#
# # Doc anh tu file
# image = cv2.imread(image_file)
# gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# # Tinh toan muc do focus cua anh
# focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
#
# if focus_measure < blur_threshold:
#     text = "Blurry pix"
#     cv2.putText(image, "{} - FM = {:.2f}".format(text, focus_measure), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#
# else:
#     text = "Fine pix"
#     cv2.putText(image, "{} - FM = {:.2f}".format(text, focus_measure), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#
# # Hien thi anh
# cv2.imshow("Image", image)
# key = cv2.waitKey(0)