from gettext import translation
import os
import sys
from typing import Type
import cv2
import argparse
import json
import numpy as np

from shutil import copyfile
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('Convert rope3D dataset to standard kitti format', add_help=False)
    parser.add_argument('--src_root', type=str, required=False, metavar="", help='root path to rope3d dataset')
    parser.add_argument('--dest_root', type=str, required=False, metavar="", help='root path to rope3d dataset in kitti format')
    args = parser.parse_args()
    return args

def copy_file(file_src, file_dest):
    if not os.path.exists(file_dest):
        try:
            copyfile(file_src, file_dest)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
            
def convert_calib(src_calib_intrinsic_file, src_calib_ltc_file, dest_calib_file):
    with open(src_calib_intrinsic_file) as f:
        intrinsic = json.load(f)
    with open(src_calib_ltc_file) as f:
        ltc = json.load(f)
    rotation = np.array(ltc['rotation'])
    translation = np.array(ltc['translation'])
    P2 = np.zeros((3, 4))
    P2[:3, :3] = np.array(intrinsic['cam_K']).reshape(3,3)
    Tr_velo_to_cam = np.concatenate((rotation, translation), axis=1)

    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P1"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P2"] = P2  # Left camera transform.
    kitti_calib["P3"] = np.zeros((3, 4))  # Dummy values.    
    kitti_calib["R0_rect"] = np.identity(3)
    kitti_calib["Tr_velo_to_cam"] = Tr_velo_to_cam
    kitti_calib["Tr_imu_to_velo"] = np.zeros((3, 4))  # Dummy values.
    
    with open(dest_calib_file, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))
            
def convert_label(src_label_file, dest_label_file):
    with open(src_label_file) as f:
        annos = json.load(f)
    new_lines = []
    for anno in annos:
        label = []
        label.append(anno['type'])
        label.append(str(anno['truncated_state']))
        label.append(str(anno['occluded_state']))
        label.append(str(anno['alpha']))
        label.append(str(anno['2d_box']['xmin']))
        label.append(str(anno['2d_box']['ymin']))
        label.append(str(anno['2d_box']['xmax']))
        label.append(str(anno['2d_box']['ymax']))
        label.append(str(anno['3d_dimensions']['h']))
        label.append(str(anno['3d_dimensions']['w']))
        label.append(str(anno['3d_dimensions']['l']))
        label.append(str(anno['3d_location']['x']))
        label.append(str(anno['3d_location']['y']))
        label.append(str(anno['3d_location']['z']))
        label.append(str(anno['rotation']))
        
        new_lines.append(' '.join(label))
    with open(dest_label_file,'w') as f:
        for line in new_lines:
            f.write(line)
            f.write("\n")

def main(src_root, dest_root):
    src_velo_dir = os.path.join(src_root, "single-vehicle-side-velodyne")
    src_image_dir = os.path.join(src_root, "single-vehicle-side-image")
    src_calib_dir = os.path.join(src_root, "single-vehicle-side", "calib")
    src_label_dir = os.path.join(src_root, "single-vehicle-side", "label", "camera")
    
    dest_velo_dir = os.path.join(dest_root, "training", "velodyne")
    dest_image_dir = os.path.join(dest_root, "training", "image_2")
    dest_calib_dir = os.path.join(dest_root, "training", "calib")
    dest_label_dir = os.path.join(dest_root, "training", "label_2")
    
    os.makedirs(dest_root, exist_ok=True)
    os.makedirs(dest_velo_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_calib_dir, exist_ok=True)
    
    for frame_name in tqdm(os.listdir(src_image_dir)):
        frame_name = frame_name.split('.')[0]
        src_velo_file = os.path.join(src_velo_dir, frame_name + '.pcd')
        src_image_file = os.path.join(src_image_dir, frame_name + '.jpg')
        src_calib_ltc_file = os.path.join(src_calib_dir, 'lidar_to_camera', frame_name + '.json')
        src_calib_intrinsic_file = os.path.join(src_calib_dir, 'camera_intrinsic', frame_name + '.json')
        src_label_file = os.path.join(src_label_dir, frame_name + '.json')
        
        dest_velo_file = os.path.join(dest_velo_dir, frame_name + '.pcd')
        dest_image_file = os.path.join(dest_image_dir, frame_name + '.png')
        dest_calib_file = os.path.join(dest_calib_dir, frame_name + '.txt')
        dest_label_file = os.path.join(dest_label_dir, frame_name + '.txt')
        
        # copy_file(src_velo_file, dest_velo_file)
        cv2.imwrite(dest_image_file, cv2.imread(src_image_file))
        copy_file(src_image_file, dest_image_file)
        convert_calib(src_calib_intrinsic_file, src_calib_ltc_file, dest_calib_file)
        convert_label(src_label_file, dest_label_file)
        
if __name__ == "__main__":
    args = parse_option()
    src_root, dest_root = args.src_root, args.dest_root
    main(src_root, dest_root)
