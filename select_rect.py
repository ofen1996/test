import argparse
import copy
import os.path
import random
# import sys

# import tifffile
import cv2
import numpy as np


def show_img(pic, name=None, line_width=3):
    global rect_x, rect_y
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, pic[y, x])
        if event == cv2.EVENT_RBUTTONDOWN:
            param[:] = [x, y]
            cv2.waitKey(300)
            cv2.destroyWindow(name)
        if event == cv2.EVENT_MOUSEMOVE:
            # n_pic = cv2.line(copy.copy(pic), (x, 0), (x, pic.shape[0]), (0, 255, 0), line_width)
            n_pic = cv2.rectangle(copy.copy(pic), (x-rect_x, y-rect_y), (x+rect_x, y+rect_y), (255, 255, 0), line_width)
            # n_pic = cv2.line(n_pic, (0, y), (pic.shape[1], y), (0, 255, 0), line_width)
            cv2.imshow(name, n_pic)

    if not name:
        name = str(random.random())
    cv2.namedWindow(name, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, (800, 600))
    cv2.imshow(name, pic)
    tmp = []
    cv2.setMouseCallback(name, mouse, tmp)
    cv2.waitKey()
    try:
        cv2.destroyWindow(name)
    except:
        pass
    return tmp[::-1]


if __name__ == '__main__':
    # print(sys.argv)
    parser = argparse.ArgumentParser(description="filter npy Area")

    parser.add_argument('--img_path', type=str, help="img which has same shape with npy")
    parser.add_argument('--npy_path', type=str, help="npy path", required=True)
    parser.add_argument('--save_dir', type=str, help="save dir", required=True)
    parser.add_argument('-x', type=int, help="rect width in (0-1000) default=100", default=100)
    parser.add_argument('-y', type=int, help="rect height in (0-1000) default=100", default=100)
    args = parser.parse_args()
    print(args)

    # img_path = r"E:\test\tmp\cell_cluster_color_outline_img.tif"
    # npy_path = r"E:\test\tmp\S_0905-LYQ-2.npy"
    img_path = args.img_path
    npy_path = args.npy_path
    save_dir = args.save_dir

    rect_x = args.x
    rect_y = args.y

    npy = np.load(npy_path)

    if img_path is not None:
        img = cv2.imread(img_path)
    else:
        img = np.zeros(npy.shape, dtype=np.uint8)
        img[npy > 0] = 200
    img_scale = cv2.resize(img, (1000, 1000))
    center_y_x = show_img(img_scale)
    print(center_y_x)

    # 画一个二值图
    mask_scale = np.zeros(img_scale.shape[:2], dtype=np.uint8)
    mask_scale[center_y_x[0]-rect_y: center_y_x[0]+rect_y, center_y_x[1]-rect_x: center_y_x[1]+rect_x] = 255
    # show_img(mask_scale)

    # 过滤细胞

    mask = cv2.resize(mask_scale, npy.shape[:2][::-1])
    npy[mask == 0] = 0

    # 过滤后npy
    img[npy == 0] = 0
    img_scale = cv2.resize(img, (1000, 1000))
    show_img(img_scale)

    filter_cell_index = list(np.unique(npy))
    filter_cell_index.remove(0)

    filter_cell_index = [f"cell_{x}" for x in filter_cell_index]

    # 保存
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "filter_cells.txt")
    with open(save_path, 'w') as f:
        f.write("\n".join(filter_cell_index))
