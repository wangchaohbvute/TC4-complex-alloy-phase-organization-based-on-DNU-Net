import os
import cv2
import numpy as np
from PIL import Image

filepath = r"PNGImages"
filename = os.listdir(filepath)
base_dir = filepath + "\\"
new_dir = r"JPEGImages\\"

for img in filename:
    '''修改图像后缀名'''
    if os.path.splitext(img)[1] == '.png' or '.PNG':
        name = os.path.splitext(img)[0]
        newFileName = name + ".jpg"
    im = cv2.imread(base_dir + img)
    im_gray1 = np.array(im)
    cv2.imwrite(new_dir + newFileName, im_gray1)
