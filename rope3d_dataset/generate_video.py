import os
import cv2
import numpy as np

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 15
    sice = (1920, 1080)
    video = cv2.VideoWriter("Instracture.mp4", fourcc, fps, sice)
    
    for id in range(1000):
        id = id + 1000
        print(id)
        image_path = "/home/yanglei/DataSets/DAIR-V2X/Coopertive-Dataset/cooperative-vehicle-infrastructure-infrastructure-side-image/" + "{:06d}".format(id) + ".jpg"
        if not os.path.exists(image_path):
            print(image_path)
            continue
        img = cv2.imread(image_path)
        video.write(img)
    video.release()
