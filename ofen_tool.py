import copy
import json
import time
import os
import time
import sys
import errno
import random
import cv2
from retry import retry


class FileLockException(Exception):
    pass


class FileLock(object):

    def __init__(self, file_name, timeout=10, delay=.05):

        self.is_locked = False
        # 将锁文件放置统一位置，方便管理
        dirs = sys.path[0] + "/lock"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        self.lockfile = os.path.join(dirs, "%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay

    def acquire(self):
        start_time = time.time()
        while True:
            try:
                # 独占式打开文件
                # os.O_RDWR : 以读写的方式打开
                # os.O_CREAT: 创建并打开一个新文件
                # os.O_EXCL: 如果指定的文件存在，返回错误
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException("Timeout occured.")
                time.sleep(self.delay)
        self.is_locked = True

    def release(self):
        # 关闭文件，删除文件
        if self.is_locked:
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False

    def __enter__(self):
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        if self.is_locked:
            self.release()

    def __del__(self):
        self.release()


def time_cost(fn):
    """
    统计耗时装饰器
    :param fn: 待装饰函数
    :return:被装饰的函数
    """
    def warp(*args, **kwargs):
        t1 = time.time()
        res = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: %s use %s" % (fn.__name__, t2 - t1))
        return res
    return warp


@retry(tries=10)
def load_json(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


@retry(tries=10)
def save_json(json_path, data):
    with FileLock(json_path, timeout=8):
        with open(json_path, "w") as fp:
            json.dumps(data)  # 先格式化，避免dump报错破坏原始文件
            json.dump(data, fp, indent=4)


def show_img(pic, name=None, line_width=3):
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, pic[y, x])
        if event == cv2.EVENT_RBUTTONDOWN:
            param[:] = [x, y]
            cv2.waitKey(300)
            cv2.destroyWindow(name)
        if event == cv2.EVENT_MOUSEMOVE:
            n_pic = cv2.line(copy.copy(pic), (x, 0), (x, pic.shape[0]), (0, 255, 0), line_width)
            n_pic = cv2.line(n_pic, (0, y), (pic.shape[1], y), (0, 255, 0), line_width)
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

def binary_pic(pic, median_blur=True):
    # floor_light = 10
    # pic = np.where(pic < floor_light, 5, pic)

    if len(pic.shape) == 3:
        B, G, R = cv2.split(pic)
        # 二值化
        B, G, R = map(binary_pic, [B, G, R])
        b_pic = cv2.merge([B, G, R])
        # show_img(b_pic)
        return b_pic

    # ret, binary = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(pic.mean())
    # show_img(pic)
    brightness = pic.mean()
    # print("brightness:{}".format(brightness))
    if brightness <= 5:
        b_threshold = -1
    elif brightness < 9:
        b_threshold = -2
    elif brightness < 15:
        b_threshold = -3
    elif brightness < 20:
        b_threshold = -4
    elif brightness < 25:
        b_threshold = -5
    else:
        b_threshold = -6
    binary = cv2.adaptiveThreshold(pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, b_threshold)
    # show_img(binary)
    if median_blur:
        binary = cv2.medianBlur(binary, ksize=3)
    # binary = cv2.medianBlur(binary, ksize=5)
    # show_img(binary)
    return binary

