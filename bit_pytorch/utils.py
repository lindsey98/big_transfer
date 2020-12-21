import xml.etree.ElementTree as ET
import re

def read_xml(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_types = []

    for boxes in root.iter('object'):

        type = boxes.find('name').text

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_all_types.append(type)
    assert len(list_with_all_boxes) == len(list_with_all_boxes)
    return list_with_all_types, list_with_all_boxes

def read_txt(txt_file: str):

    contents = [x.strip() for x in open(txt_file).readlines()]
    paths = [x.split('\t')[0] for x in contents]
    classes = [x.split('\t')[2] for x in contents]
    labels = [x.split('\t')[-1] for x in contents]
    coordinates = [x.split('\t')[1] for x in contents]
    preprocess_coordinates = []

    for coord in coordinates:
        x1, y1, x2, y2 = list(map(float, re.search(r'\((.*?)\)', coord).group(1).split(",")))
        preprocess_coordinates.append([x1, y1, x2, y2])

    assert (len(preprocess_coordinates) == len(classes)) & (len(paths) == len(preprocess_coordinates)) & (len(labels) == len(classes))
    return labels, paths, preprocess_coordinates, classes
