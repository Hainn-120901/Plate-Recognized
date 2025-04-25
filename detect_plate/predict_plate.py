from ultralytics import YOLO
from detect_plate.process_plate import crop_image, calculate_iou_for_boxes, calculate_bbox_area

class PlateDetector:
    def __init__(self):
        self.conf_threshold: float = 0.2
        self.model_path = "weight/detect_plate/detect_plate.pt"
        self.model = YOLO(self.model_path)

    def predict(self, image):

        results = self.model(image)[0]  # Lấy kết quả từ YOLO
        list_box_text = []
        list_score_text = []

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if conf > self.conf_threshold:
                list_box_text.append([int(x1), int(y1), int(x2), int(y2)])
                list_score_text.append(conf)

        list_box_text = sorted(list_box_text, key=lambda x:-(calculate_bbox_area(x)))

        list_box_all = calculate_iou_for_boxes(list_box_text)
        print(list_box_text)
        all_image = crop_image(image, list_box_all)
        return all_image
