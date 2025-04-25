from detect_plate.predict_plate import PlateDetector
from detect_conner.predict_align import AlignPlates
from detect_text.predict_text import TextPlates
from recognized_text.recognized_text import RecognizedText
import cv2
from PIL import Image

class LicensePlate():
    def __init__(self):
        self.plate_detector = PlateDetector()
        self.align_plates = AlignPlates()
        self.text_plates = TextPlates()
        self.recognized_text = RecognizedText()

    def predict_plate(self, image):
        all_image_plate = self.plate_detector.predict(image)
        if len(all_image_plate) == 0:
            return [], []
        list_result = []
        list_image_align = []
        for image_plate in all_image_plate:
            image_align = self.align_plates.predict_align(image_plate)
            list_cropped_images_all, list_score_all, list_box_all = self.text_plates.predict_text(image_align)
            if list_cropped_images_all is not None:
                l_res_all = []
                for idx_list_crop_image, list_cropped_images in enumerate(list_cropped_images_all):
                    l_res_in_line = []
                    for cropped_images in list_cropped_images:
                        img_color = cv2.cvtColor(cropped_images, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_color)
                        result_text_single = self.recognized_text.reconized(img_pil)
                        l_res_in_line.append(result_text_single)
                    if idx_list_crop_image == 0:
                        if len(l_res_in_line) == 2:
                            result_text_in_line = "-".join(l_res_in_line)
                        elif len(l_res_in_line) == 3:
                            result_text_in_line = ".".join(["-".join([l_res_in_line[0], l_res_in_line[1]]), l_res_in_line[2]])
                        else:
                            result_text_in_line = " ".join(l_res_in_line)
                    elif idx_list_crop_image == 1:
                        result_text_in_line = ".".join(l_res_in_line)
                    else:
                        result_text_in_line = " ".join(l_res_in_line)
                    l_res_all.append(result_text_in_line)
                result_text_all = " ".join(l_res_all)
                list_result.append(result_text_all)
                list_image_align.append(image_align)
        return list_result, list_image_align
