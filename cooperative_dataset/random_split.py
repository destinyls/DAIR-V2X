import os
import argparse
import random

def parse_option():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--root', type=str, required=False, metavar="", help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    root = args.root
    
    os.makedirs(os.path.join(root, "ImageSets"), exist_ok=True)
    train_txt = os.path.join(root, "ImageSets", "train.txt")
    val_txt = os.path.join(root, "ImageSets", "val.txt")
    trainval_txt = os.path.join(root, "ImageSets", "trainval.txt")

    trainval_list = []
    for idx in range(20000):
        if os.path.exists(os.path.join(root, "training", "denorm", "{:06d}".format(idx)) + ".txt"):
            trainval_list.append(idx)
    
    with open(trainval_txt,'w') as f:
        for idx in trainval_list:
            f.write("{:06d}".format(idx))
            f.write("\n")    
    
    random.shuffle(trainval_list)
    train_list = trainval_list[:int(0.7 * len(trainval_list))]
    val_list = trainval_list[int(0.7 * len(trainval_list)):]

    with open(train_txt,'w') as f:
        for idx in train_list:
            f.write("{:06d}".format(idx))
            f.write("\n")
    with open(val_txt,'w') as f:
        for idx in val_list:
            f.write("{:06d}".format(idx))
            f.write("\n")