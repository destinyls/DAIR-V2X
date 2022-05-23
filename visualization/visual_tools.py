import os
import math
import cv2
import argparse
import numpy as np

from PIL import Image
from visual_utils import *

def parse_option():
    parser = argparse.ArgumentParser('Visualize InfrastructureSide dataset', add_help=False)
    parser.add_argument('--root_dir', type=str, default='datasets/Rope3D-KITTI', help='root path to InfrastructureSide dataset')
    parser.add_argument('--pred_dir', type=str, default='', help='path to preds')    
    parser.add_argument('--split', type=str, default='val', help='split: train or val',)
    args = parser.parse_args()
    return args

class InfrastructureSide():
    def __init__(self, root_dir, split, pred_dir):        
        # path configuration
        self.data_dir = os.path.join(root_dir, 'training' if split == 'val' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.denorm_dir = os.path.join(self.data_dir, 'denorm')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        
        split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]
        
        self.has_preds = False
        if pred_dir != '':
            self.has_preds = True
            self.pred_dir = pred_dir
            
    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)      # (H, W, 3) RGB mode

    def get_label(self, idx, get_preds=False):
        if get_preds:
            label_file = os.path.join(self.pred_dir, '%06d.txt' % idx)
        else:
            label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)
    
    def get_calib(self, idx):
            calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
            assert os.path.exists(calib_file)
            return Calibration(calib_file)
        
    def get_denorm(self, idx):
        denorm_file = os.path.join(self.denorm_dir, '%06d.txt' % idx)
        assert os.path.exists(denorm_file)
        with open(denorm_file, 'r') as f:
            lines = f.readlines()
        denorm = np.array([float(item) for item in lines[0].split(' ')])
        return denorm
    
    def __getitem__(self, item):
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        
        demo_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)    
        denorm = self.get_denorm(index)
        calib = self.get_calib(index)
        objects = self.get_label(index)
        object_num = len(objects)
        
        for i in range(object_num):
            print(objects[i].level_str, objects[i].pos[-1])
            if objects[i].level_str == 'UnKnown':
                continue
            _, corners3d = objects[i].generate_corners3d_denorm(denorm)     # real 3D center in 3D space
            pts_img, _ = calib.rect_to_img(corners3d)
            demo_img = draw_box_3d(demo_img, pts_img, (0, 0, 255))
            box2d = objects[i].box2d
            demo_img = cv2.rectangle(demo_img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (255, 255, 0), 2)

        if self.has_preds:
            objects = self.get_label(index, True)
            for i in range(len(objects)):
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue
                _, corners3d = objects[i].generate_corners3d_denorm(denorm)     # real 3D center in 3D space
                pts_img, _ = calib.rect_to_img(corners3d)
                demo_img = draw_box_3d(demo_img, pts_img, (0, 255, 0))
                box2d = objects[i].box2d
                demo_img = cv2.rectangle(demo_img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (255, 0, 0), 2)
        cv2.imwrite(os.path.join("debug", str(index) + ".jpg"), demo_img)
        
if __name__ == "__main__":
    args = parse_option()
    dataset = InfrastructureSide(args.root_dir, args.split, args.pred_dir)
    os.makedirs("debug", exist_ok=True)

    for i in range(100):
        dataset[i]
