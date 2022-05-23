import argparse
import os
import csv
import cv2
import numpy as np

from PIL import Image

TYPE_ID_CONVERSION = {
    'Car': 0, 
    'Van': 1,
    'Truck': 2, 
    'Bus': 3, 
    'Pedestrian': 4, 
    'Cyclist': 5, 
    'Motorcyclist': 6, 
    'Tricyclist': 7, 
    'Trafficcone': 8,
    'Unknown_unmovable': 9,
    'Barrow': 10,
    'Unknowns_movable': 11,
    'Triangle plate': 12
}

def parse_option():
    parser = argparse.ArgumentParser('Background Elimination Tools', add_help=False)
    parser.add_argument('--root', type=str, default='', help='root path to InfrastructureSide dataset in KITTI format')
    args = parser.parse_args()
    return args

class BackgroundElimination():
    def __init__(self, root_dir):        
        self.data_dir = os.path.join(root_dir, 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.denorm_dir = os.path.join(self.data_dir, 'denorm')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        
        split_dir = os.path.join(root_dir, 'ImageSets', 'trainval.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]
        
        self.template_maske = np.zeros((1080, 1920, 3))
        self.classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist', 'Motorcyclist', 'Tricyclist', 'Trafficcone', 'Unknown_unmovable', 'Barrow', 'Unknowns_movable', 'Triangle plate']

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, self.idx_list[idx] + ".png")
        assert os.path.exists(img_file)
        return cv2.imread(img_file)      # (H, W, 3) RGB mode
    
    def get_annos(self, idx):
        annos = []
        file_name = self.idx_list[idx] + ".txt"
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
        with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                if row["type"] in self.classes:
                    annos.append({
                        "class": row["type"],
                        "label": TYPE_ID_CONVERSION[row["type"]],
                        "truncation": float(row["truncated"]),
                        "occlusion": float(row["occluded"]),
                        "alpha": float(row["alpha"]),
                        "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                        "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                        "rot_y": float(row["ry"])
                    })
        return annos
    
    def process(self, frames=100):
        for idx in range(frames):
            image = self.get_image(idx)
            annos = self.get_annos(idx)
            print(image.shape)
    
if __name__ == "__main__":
    args = parse_option()
    back_e = BackgroundElimination(args.root)
    back_e.process(100)
    
    
    
    