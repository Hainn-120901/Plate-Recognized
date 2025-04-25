import numpy as np


def sort_box(filtered_box):
    sortted_xmin = sorted(filtered_box, key=lambda box: box[1])
    lines = []

    if len(sortted_xmin) > 0:
        first_box = sortted_xmin[0]
        ymin = first_box[1]
        ymax = first_box[3]
        y_center = (ymin + ymax) / 2
        lines.append([first_box])

        for box in sortted_xmin[1:]:
            ymin = box[1]
            ymax = box[3]

            if ymin <= y_center:
                lines[-1].append(box)
            else:
                lines.append([box])
                y_center = (ymin + ymax) / 2

    lines_final = []
    for line in lines:
        line = sorted(line, key=lambda x: x[0])
        lines_final.append(line)
    return lines_final


def crop_image(im1, list_box_rs_all):
    cropped_images_all = []
    list_score_all = []
    list_box_all = []
    for box_same_line in list_box_rs_all:
        cropped_images = []
        list_score = []
        list_box = []
        for box in box_same_line:
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_image = im1[ymin:ymax, xmin:xmax]
            cropped_images.append(cropped_image)
            list_score.append(float(box[4]))
            list_box.append([xmin, ymin, xmax, ymax])
        cropped_images_all.append(cropped_images)
        list_box_all.append(list_box)
        list_score_all.append(list_score)
    return cropped_images_all, list_score_all, list_box_all


def crop_image_v2(im1, list_box_rs_all, key):
    xmin_final, ymin_final, xmax_final, ymax_final = [], [], [], []
    list_image_crop = []
    list_score_final = []
    for i in range(len(list_box_rs_all)):  # loai_label
        xmin_, ymin_, xmax_, ymax_ = [], [], [], []
        list_image = []
        list_score = []
        for boxx in list_box_rs_all[i]:
            xmin, xmax, ymin, ymax = None, None, None, None
            cropped_image = None
            if key == 1:
                xmin, ymin, xmax, ymax = int(boxx[0] - 1), int(boxx[1]), int(boxx[2] + 4), int(boxx[3] + 2)
            if key == 2:
                xmin, ymin, xmax, ymax = int(boxx[0] - 1), int(boxx[1]), int(boxx[2] + 4), int(boxx[3] + 3)
            if key == 3:
                xmin, ymin, xmax, ymax = int(boxx[0]), int(boxx[1]), int(boxx[2] + 4), int(boxx[3] + 3)
            if key == 4:
                xmin, ymin, xmax, ymax = int(boxx[0]), int(boxx[1] + 1), int(boxx[2] + 3), int(boxx[3] + 2)
            if key == 5:
                xmin, ymin, xmax, ymax = int(boxx[0]), int(boxx[1]), int(boxx[2] + 3), int(boxx[3] + 2)
            if key == 6:
                xmin, ymin, xmax, ymax = int(boxx[0]), int(boxx[1] + 1), int(boxx[2] + 4), int(boxx[3] + 2)
            if key == 7:
                xmin, ymin, xmax, ymax = int(boxx[0]), int(boxx[1] + 1), int(boxx[2] + 4), int(boxx[3] + 4)
            if key == 8:
                xmin, ymin, xmax, ymax = int(boxx[0]), int(boxx[1]), int(boxx[2] + 4), int(boxx[3] + 2)
            list_score.append(float(boxx[4]))
            if xmin is not None:
                cropped_image = im1[ymin:ymax, xmin:xmax]
            if cropped_image is not None:
                list_image.append(cropped_image)
            xmin_.append(xmin)
            ymin_.append(ymin)
            xmax_.append(xmax)
            ymax_.append(ymax)
        if len(list_image) > 0:
            list_image_crop.extend(list_image)
            xmin_final.extend(xmin_)
            ymin_final.extend(ymin_)
            xmax_final.extend(xmax_)
            ymax_final.extend(ymax_)
            list_score_final.extend(list_score)
    if len(ymin_final) > 0 and len(list_score_final) > 0:
        box_ = [min(xmin_final), min(ymin_final), max(xmax_final), max(ymax_final)]
        score_ = np.mean(list_score_final)
        return list_image_crop, box_, score_
    return None, None, None

