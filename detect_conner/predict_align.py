from ultralytics import YOLO
from detect_conner.process_image import align_image
class AlignPlates:
    def __init__(self):
        self.model_path = "weight/detect_conner/weight_detect_conner.pt"
        self.device = "cpu"
        self.model = YOLO(self.model_path).to(self.device)
        self.list_class = {0: "top_left", 1: "top_right", 2: "bottom_left", 3: "bottom_right"}
        self.thresh = 0.2

    def predict_align(self, image):
        results = self.model(image)[0]  # Lấy kết quả từ YOLO
        dict_conner = {}

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if conf > self.thresh:
                if int(cls) in list(dict_conner.keys()):
                    if conf > dict_conner[int(cls)][4]:
                        dict_conner[int(cls)] = [x1, y1, x2, y2, conf]
                    else:
                        continue
                else:
                    dict_conner[int(cls)] = [x1, y1, x2, y2, conf]
        # Align image
        cropped_img = align_image(image, dict_conner)
        return cropped_img if cropped_img is not None else image
