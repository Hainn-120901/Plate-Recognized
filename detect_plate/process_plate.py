def crop_image(im1, list_box_rs_all):
    cropped_images_all = []
    for box in list_box_rs_all:
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_image = im1[ymin:ymax, xmin:xmax]
        cropped_images_all.append(cropped_image)

    return cropped_images_all

def calculate_iou(box1, box2):
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    intersection = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)

    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area_box1 + area_box2 - intersection

    iou = intersection / union

    return iou


def merge_boxes(box1, box2):
    xmin = min(box1[0], box2[0])
    ymin = min(box1[1], box2[1])
    xmax = max(box1[2], box2[2])
    ymax = max(box1[3], box2[3])

    merged_box = [xmin, ymin, xmax, ymax]
    return merged_box


def calculate_iou_for_boxes(boxes):
    l_box_total = []

    for bx in boxes:
        if len(l_box_total) == 0:
            l_box_total.append(bx)
        else:
            for idx_b, b in enumerate(l_box_total):
                iou = calculate_iou(bx, b)
                if iou >= 0.2:
                    merged_box = merge_boxes(b, bx)
                    l_box_total[idx_b] = merged_box
    return l_box_total

def calculate_bbox_area(box):
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
    width = x_max - x_min
    height = y_max - y_min
    return width * height if width > 0 and height > 0 else 0