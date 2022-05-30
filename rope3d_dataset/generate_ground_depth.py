import os
import argparse
import csv
import cv2
import time
import numpy as np

from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--kitti_root', type=str, required=False, metavar="", help='')
    args = parser.parse_args()
    return args

def load_denorm(denorm_file):
    assert os.path.exists(denorm_file)
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def calCoordinateFrom2PointsAndPlane(P2, denorm):    
    pos = -1 * P2 * denorm[3] / np.inner(denorm[:3], np.array(P2)).reshape(P2.shape[0], 1)
    return pos 

def decode_location(P, point2d, depth):
    P = np.vstack((P, [0.0,0.0,0.0,1.0]))
    P_inv = np.linalg.pinv(P)
    point_extend = np.hstack([point2d[:, 0].reshape(point2d.shape[0], 1), point2d[:, 1].reshape(point2d.shape[0], 1), np.ones(point2d.shape[0]).reshape(point2d.shape[0], 1)])
    point_extend = point_extend * depth
    
    point_extend = np.hstack([point_extend, np.ones(point_extend.shape[0]).reshape(point_extend.shape[0], 1)])
    locations = np.matmul(P_inv, point_extend.T)
    locations = locations[:3]
    return locations

def load_P(calib_file):
    with open(calib_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                break
    return  P2

def generate_depth(P, denorm, point2d):
    locs = decode_location(P, point2d, 10).T
    ground_locs = calCoordinateFrom2PointsAndPlane(locs, denorm)
    depths = ground_locs[:, 2].reshape(1080, 1920, 1)
    depths = np.clip(depths, 0, 255)
    
    return depths


if __name__ == "__main__":
    args = parse_option()
    
    kitti_root = args.kitti_root
    calib_dir = os.path.join(kitti_root, "training", "calib")
    denorm_dir = os.path.join(kitti_root, "training", "denorm")
    depth_equation_dir = os.path.join(kitti_root, "training", "depth_equation")
    os.makedirs(depth_equation_dir, exist_ok=True)

    xx1, yy1 = np.meshgrid(np.arange(0, 1920, 1), np.arange(0, 1080, 1))
    xx1 = xx1.reshape((1080, 1920, 1))
    yy1 = yy1.reshape((1080, 1920, 1))
    point2d = np.concatenate((xx1, yy1), axis=2)
    point2d = point2d.reshape(-1, 2)   
    
    for frame_name in tqdm(os.listdir(denorm_dir)):
        frame_name = frame_name.split('.')[0]
        denorm_file = os.path.join(denorm_dir, frame_name + ".txt")
        denorm = load_denorm(denorm_file)
        calib_file = os.path.join(calib_dir, frame_name + ".txt")
        P = load_P(calib_file)
        depths = generate_depth(P, denorm, point2d)

        depth_equation_file = os.path.join(depth_equation_dir, frame_name + ".jpg")
        cv2.imwrite(depth_equation_file, depths)
        
        
                
                