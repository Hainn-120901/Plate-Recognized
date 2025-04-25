import torch, cv2, os
from lxml import etree

def create_pascal_voc_xml(self, image_path, output_xml_path):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    annotation = etree.Element("annotation")
    folder = etree.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(image_path))

    filename = etree.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(image_width)
    etree.SubElement(size, "height").text = str(image_height)
    etree.SubElement(size, "depth").text = "3"

    results = self.model(image)[0]

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        class_name = str(int(cls))  # Lớp đối tượng
        x_min = int(x1)
        y_min = int(y1)
        x_max = int(x2)
        y_max = int(y2)

        obj = etree.SubElement(annotation, "object")
        etree.SubElement(obj, "name").text = class_name
        bndbox = etree.SubElement(obj, "bndbox")
        etree.SubElement(bndbox, "xmin").text = str(x_min)
        etree.SubElement(bndbox, "ymin").text = str(y_min)
        etree.SubElement(bndbox, "xmax").text = str(x_max)
        etree.SubElement(bndbox, "ymax").text = str(y_max)

    tree = etree.ElementTree(annotation)
    tree.write(output_xml_path)