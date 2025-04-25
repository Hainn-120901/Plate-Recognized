import cv2

from license_plates import LicensePlate

model = LicensePlate()
path_image = "/home/teamai/Downloads/729a0cb058474522bf32eb0ea59202ee.jpg"
path_save = "test"
image_ori = cv2.imread(path_image)
result = model.predict_plate(image_ori)
print(result)