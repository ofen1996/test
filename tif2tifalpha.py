import tifffile
import os
import cv2
import numpy as np


def imread(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def add_alpha_channl(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_with_alpha = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    gray = np.where(gray < 15, 0, gray)
    gray[gray > 220] = 0
    gray[gray != 0] = 255

    img_with_alpha[..., 3] = gray
    return img_with_alpha


if __name__ == '__main__':
    base_dir = r"G:\细胞分割素材"
    img_dirs = [os.path.join(base_dir, dirs) for dirs in os.listdir(base_dir)]
    for img_dir in img_dirs:
        # img_dir = r"G:\细胞分割素材\FG4 小鼠睾丸"
        save_dir = os.path.join(img_dir, "去背景")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        img_names = [name for name in os.listdir(img_dir) if name.endswith(".tif") or name.endswith(".png")]
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            img = imread(img_path)
            if img.shape[-1] == 4:
                print(f"{img_path} is already with alpha, skip it")
                continue
            img_with_alpha = add_alpha_channl(img)
            save_name = os.path.splitext(img_name)[0]
            save_path = os.path.join(save_dir, save_name + ".tif")
            print(f"{save_path} start save")
            tifffile.imwrite(save_path, img_with_alpha, compression='lzw')

