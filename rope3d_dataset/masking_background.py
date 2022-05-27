import os
import argparse
import csv
import cv2
import numpy as np
from tqdm import tqdm

category_map = ['Car', 'Big_Vehicle', 'Pedestrian', 'Cyclist', 'Trafficcone', 'Unknown_unmovable', 'Unknowns_movable', 'Triangle plate']

def parse_option():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--kitti_root', type=str, required=False, metavar="", help='')
    args = parser.parse_args()
    return args

def load_boxes(label_file):
    boxes = []
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                  'dl', 'lx', 'ly', 'lz', 'ry']
    with open(label_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row['type'] not in category_map:
                continue
            box = [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            boxes.append(box)
    return boxes

def remove_background(image, label_file):
    empty_image = np.ones_like(image) * 125
    boxes = load_boxes(label_file)
    for box in boxes:
        empty_image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
    return empty_image   
   
if __name__ == "__main__":
    args = parse_option()
    kitti_root = args.kitti_root
    
    label_path = os.path.join(kitti_root, "training", "label_2")
    image_path = os.path.join(kitti_root, "training", "image_2")
    image_masked_path = os.path.join(kitti_root, "training", "image_2_masked")
    os.makedirs(image_masked_path, exist_ok=True)
    
    for label_name in tqdm(os.listdir(label_path)):
        frame_name = label_name.split('.')[0]
        label_file = os.path.join(label_path, frame_name + ".txt")
        image_file = os.path.join(image_path, frame_name + ".png")
        image_masked_file = os.path.join(image_masked_path, frame_name + ".png")
        
        image = cv2.imread(image_file)
        masked_image = remove_background(image, label_file)
        cv2.imwrite(image_masked_file, masked_image)
