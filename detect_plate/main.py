import cv2
from predict_plate import PlateDetector

if __name__ == "__main__":
    image_path = "/home/teamai/Downloads/62623.jpg"

    image = cv2.imread(image_path)
    detector = PlateDetector()

    # Dự đoán và in kết quả
    _, list_boxes, list_scores = detector.predict(image)
    # print("Bounding Boxes:", boxes)
    # print("Confidence Scores:", scores)
    for boxes in list_boxes:
        cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
    cv2.imshow("test", image)
    cv2.waitKey(0)

