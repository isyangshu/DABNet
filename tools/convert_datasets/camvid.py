import os
import cv2
import mmcv
import numpy as np
from PIL import Image

target_path = '/workdir/DABNet/data/CamVid/TrainID/val'
original_path = '/workdir/DABNet/data/CamVid/annotations/val'
PALETTE = [[128, 128, 128],
           [128, 0, 0],
           [192, 192, 128],
           [128, 64, 128],
           [0, 0, 192],
           [128, 128, 0],
           [192, 128, 128],
           [64, 64, 128],
           [64, 0, 128],
           [64, 64, 0],
           [0, 128, 192]]

file_list = os.listdir(original_path)

for filename in file_list:
    filename_ = filename.split('.')[0] + '_TrainID.png'
    filename = os.path.join(original_path, filename)
    targetname = os.path.join(target_path, filename_)

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    TrainId = np.ones((720, 960))*255
    for id, p in enumerate(PALETTE):
        for i in range(720):
            for j in range(960):
                if image[i,j,0] == p[0] and image[i,j,1] == p[1] and image[i,j,2]==p[2]:
                    TrainId[i,j] = id
    cv2.imwrite(targetname, TrainId)