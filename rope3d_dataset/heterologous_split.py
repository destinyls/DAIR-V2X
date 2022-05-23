import os
import argparse
import random
import json

def parse_option():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--root', type=str, required=False, metavar="", help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    root = args.root
    train_json = os.path.join(root, "ImageSets", "train.json")
    val_json = os.path.join(root, "ImageSets", "val.json")
    
    het_train_txt = os.path.join(root, "ImageSets", "het_train.txt")
    het_val_txt = os.path.join(root, "ImageSets", "het_val.txt")
    
    with open(train_json, 'r') as f:
        train_dict = json.load(f)
    with open(val_json, 'r') as f:
        val_dict = json.load(f)
        
    trainval_dict = dict()
    for k, v in train_dict.items():
        trainval_dict[k] = v
    for k, v in val_dict.items():
        trainval_dict[k] = v
            
    scene_list = dict()
    scene_name_list = []
    for idx_str in trainval_dict.keys():
        idx = idx_str.split('_')
        scene_name = idx[1]
        if scene_name not in scene_list.keys():
            scene_list[scene_name] = [trainval_dict[idx_str]]
            scene_name_list.append(scene_name)
        else:
            scene_list[scene_name].append(trainval_dict[idx_str])

    train_list, val_list = [], []
    random.shuffle(scene_name_list)
    for scene_name in scene_name_list[6:]:
        train_list = train_list + scene_list[scene_name]
    for scene_name in scene_name_list[:6]:
        val_list = val_list + scene_list[scene_name]
        
    print(len(train_list), len(val_list))
    
    with open(het_train_txt, 'w') as f:
        for idx in train_list:
            f.write(idx)
            f.write("\n")
    with open(het_val_txt, 'w') as f:
        for idx in val_list:
            f.write(idx)
            f.write("\n")