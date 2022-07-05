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
    train_txt = os.path.join(root, "ImageSets", "train.txt")
    val_txt = os.path.join(root, "ImageSets", "val.txt")
    trainval_txt = os.path.join(root, "ImageSets", "trainval.txt")
    
    hom_train_txt = os.path.join(root, "ImageSets", "hom_train.txt")
    hom_val_txt = os.path.join(root, "ImageSets", "hom_val.txt")
    
    train_list = [x.strip() for x in open(train_txt).readlines()]
    val_list = [x.strip() for x in open(val_txt).readlines()]
    trainval_list = train_list + val_list
    
    with open(trainval_txt,'w') as f:
        for idx in trainval_list:
            f.write(idx)
            f.write("\n")    
    # os.remove(train_txt)
    # os.remove(val_txt)
    
    random.shuffle(trainval_list)
    train_list = trainval_list[:int(0.7 * len(trainval_list))]
    val_list = trainval_list[int(0.7 * len(trainval_list)):]
      
    with open(hom_train_txt,'w') as f:
        for idx in train_list:
            f.write(idx)
            f.write("\n")
    with open(hom_val_txt,'w') as f:
        for idx in val_list:
            f.write(idx)
            f.write("\n")