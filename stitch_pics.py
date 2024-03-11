import numpy as np
from PIL import Image
import os
import tifffile

## 郭春明老师的程序

pics_dir = r"E:\test\20231222_153155.sdpc 38_32 142_142 1"
row = 38
col = 32

pixes = np.array((2048, 2448))
de_pixes = pixes // 2

whole_size = (de_pixes[0] * row, de_pixes[1] * col, 3)
whole_img = np.zeros(whole_size, dtype=np.uint8)
for c in range(col):
    for r in range(row):
        pic_name = "{}_{}_00.jpg".format(str(col-1-c).zfill(3), str(r).zfill(4))
        print(pic_name)
        img = np.asarray(Image.open(os.path.join(pics_dir, pic_name)))
        whole_img[r*de_pixes[0]: (r+1)*de_pixes[0], c*de_pixes[1]: (c+1)*de_pixes[1]] = img[::2, ::2, ...]


tifffile.imwrite(os.path.join(pics_dir, "whole_img.tif"), whole_img, compression="jpeg")