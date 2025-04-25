from ultralytics import YOLO
from detect_text.image_process import sort_box, crop_image

class TextPlates:
    def __init__(self):
        self.model_path = "weight/detect_text/weight_detect_text.pt"
        self.device = "cpu"
        self.model = YOLO(self.model_path).to(self.device)
        self.list_class = {0: "text"}
        self.thresh = 0.3

    def predict_text(self, image):
        results = self.model(image)[0]  # Lấy kết quả từ YOLO
        list_box_text = []

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if conf > self.thresh:
                 list_box_text.append([int(x1), int(y1), int(x2), int(y2), conf])
        if len(list_box_text) > 0:
            list_box_sort = sort_box(list_box_text)
            cropped_images, list_score, list_box = crop_image(image, list_box_sort)
            return cropped_images, list_score, list_box
        else:
            return None, None, None

