import os
import argparse
import random
import numpy as np

category_map = {'Car': 'Car', 'Van': 'Car', 'Truck': 'Big_Vehicle', 'Bus': 'Big_Vehicle', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist', 'Motorcyclist': 'Cyclist', 'Tricyclist': 'Cyclist', 'Trafficcone': 'Trafficcone', 'Unknown_unmovable': 'Unknown_unmovable', 'Barrow': 'Cyclist', 'Unknowns_movable': 'Unknowns_movable', 'Triangle plate': 'Triangle plate'}

def parse_option():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--src_path', type=str, required=False, metavar="", help='')
    parser.add_argument('--dest_path', type=str, required=False, metavar="", help='')
    args = parser.parse_args()
    return args
    
def convert_label(src_label_file, dest_label_file):
    with open(src_label_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        label = line.strip().split(' ')    
        pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)    
        if np.sum(pos) < 10e-9:
            continue
        
        label[0] = category_map[label[0]]
        label[1] = str(round(float(label[1]), 2)) 
        label[3] = str(round(float(label[3]), 2))
        label[4] = str(round(float(label[4]), 2))
        label[5] = str(round(float(label[5]), 2))
        label[6] = str(round(float(label[6]), 2))
        label[7] = str(round(float(label[7]), 2))
        label[8] = str(round(float(label[8]), 2))
        label[9] = str(round(float(label[9]), 2))
        label[10] = str(round(float(label[10]), 2))
        label[11] = str(round(float(label[11]), 2))
        label[12] = str(round(float(label[12]), 2))
        label[13] = str(round(float(label[13]), 2))
        label[14] = str(round(float(label[14]), 2))
        new_lines.append(' '.join(label))
        
    with open(dest_label_file,'w') as f:
        for line in new_lines:
            f.write(line)
            f.write("\n")
            
if __name__ == "__main__":
    args = parse_option()
    src_path, dest_path = args.src_path, args.dest_path
    for label_name in os.listdir(src_path):
        convert_label(os.path.join(src_path, label_name), os.path.join(dest_path, label_name))
        print(label_name)