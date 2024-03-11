import cv2
import tifffile
from matplotlib import pyplot as plt
import json
import getopt
import sys
import os
from ofen_tool import show_img
ShowImageType = 1

def cv_show(name, img):
    if ShowImageType == 0:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.imshow(img)
        plt.title(name)
        plt.show()


# -----------------------------------------------------------------
# 荧光图片预处理
# -----------------------------------------------------------------
def fl_img_process(img):
    img_fl = img
    gray = img_fl[..., 1] if len(img_fl.shape) == 3 else img_fl

    kernel_size = img_fl.shape[0] // 2000
    kernel_size = 3 if kernel_size < 3 else kernel_size  # 限制最小

    gray = cv2.blur(gray, (kernel_size, kernel_size))
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # show_img(binary)
    img_dist = erode_dilate_process(binary, [['e', 3, 2], ['d', kernel_size, 8], ['e', kernel_size, 5], ['d', kernel_size, 9]])
    return img_dist


# -----------------------------------------------------------------
# HE图片预处理
# -----------------------------------------------------------------
def he_img_process(img, gray_num):
    # 灰度处理
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img[:, :, 1]  # 灰度处理改为选取差异较大的通道 2022.9.28 欧阳峰修改
    # 二值化处理
    ret, binary = cv2.threshold(gray, gray_num, 255, cv2.THRESH_BINARY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    show_img(binary)
    # 黑白反转，将HE区域调成亮色
    binary = 255 - binary
    show_img(binary)
    # OpenCV定义的结构元素
    kernel_size = img.shape[0] // 2000
    kernel_size = 3 if kernel_size < 3 else kernel_size  # 限制最小
    img_dist = erode_dilate_process(binary,
                                    [['e', kernel_size, 1], ['d', kernel_size, 8], ['e', kernel_size, 10], ['d', kernel_size, 6]])
    # # 腐蚀图像
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # eroded = cv2.erode(binary, kernel3, iterations=2)
    # # cv_show("eroded Image", eroded)
    #
    # # 膨胀图像
    # kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilated2 = cv2.dilate(eroded, kernel5, iterations=13)
    #
    # # 腐蚀图像
    # eroded2 = cv2.erode(dilated2, kernel3, iterations=15)
    # # cv_show("eroded2 Image", eroded2)
    #
    # # 膨胀图像
    # dilated3 = cv2.dilate(eroded2, kernel5, iterations=1)
    # # cv_show("dilated3 Image", dilated3)

    return img_dist





# -----------------------------------------------------------------
# HE轮廓识别
# 过程：先识别全部的轮廓，然后进行轮廓过滤
# 1) 过滤面积过小的轮廓
# 2) 过滤子轮廓
# -----------------------------------------------------------------
def search_he_region(binary_img):
    # 轮廓识别
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 轮廓过滤
    filter_conts = []
    # 即面积小于一个BLOCK大小1/2的轮廓都会被过滤掉
    area_cutoff = 1.0 * binary_img.shape[0] * binary_img.shape[1] / 46 / 46 / 2
    area_sum = 0.0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area <= area_cutoff:
            continue
        if hierarchy[0][i][3] != -1:
            continue
        filter_conts.append(contours[i])
        area_sum += area

    return area_sum, filter_conts
    pass


# -----------------------------------------------------------------
# roi input output
# -----------------------------------------------------------------
def save_roi_2_file(outfile, conts):
    # 格式转化
    jconts = []
    for i in range(len(conts)):
        cont = conts[i]
        curr_cont = []
        for c in cont:
            curr_cont.append([int(c[0][0]),int(c[0][1])])
        jconts.append(curr_cont)

    # output
    json_dict = {}
    json_dict['ori_group'] = jconts
    with open(outfile, "w") as fp:
        json.dump(json_dict, fp, indent=4)
    pass


def write_stat_2_file(outfile, img, area_sum):
    with open(outfile, 'w') as fp:
        fp.write(f'height\t{img.shape[0]}\n')
        fp.write(f'width\t{img.shape[1]}\n')
        fp.write(f'cover_rate\t{area_sum/img.shape[0]/img.shape[1]}\n')


def get_img_conts(img, fluorescence="fl", gray_num=240):
    # 图片预处理
    if fluorescence == 'fl':
        img_binary = fl_img_process(img)
    else:
        img_binary = he_img_process(img, gray_num)
    show_img(img_binary)
    # HE区域识别
    area_sum, filter_conts = search_he_region(img_binary)
    return filter_conts


def usage():
    print("======="*10)
    print("Usage:")
    print("1) python levelMatrix.py -i ./test.tif -o ./outdir/")
    print("2) python levelMatrix.py --infile ./test.tif --outdir ./outdir/")
    print("3) python levelMatrix.py --infile ./test.tif --outdir ./outdir/ -g 185")
    print("3) python levelMatrix.py --infile ./fl.tif --outdir ./outdir/ -f 表示荧光图像")
    print("4) python levelMatrix.py -h")
    print("5) python levelMatrix.py --help")
    print("======="*10)
    exit()


def main():
    """
    getopt 模块的用法
    """
    options, args = getopt.getopt(sys.argv[1:], 'hi:o:g:f', ['help', 'infile=', 'outdir=', 'gray='])

    # init
    indir = None
    outdir = None
    gray_num = 200
    fluorescence = False

    options_dict = {}
    for name, value in options:
        if name not in options_dict.keys():
            options_dict[name] = value

    if '-h' in options_dict.keys() or '--help' in options_dict.keys():
        usage()

    if '-i' not in options_dict.keys() and '--infile' not in options_dict.keys():
        print("-i or --infile must exist")
        usage()

    if '-o' not in options_dict.keys() and '--outdir' not in options_dict.keys():
        print("-o or --outdir must exist")

    if '-i' in options_dict.keys():
        infile = options_dict['-i']
    if '--infile' in options_dict.keys():
        infile = options_dict['--infile']

    if '-o' in options_dict.keys():
        outdir = options_dict['-o']
    if '--outdir' in options_dict.keys():
        outdir = options_dict['--outdir']

    if '-g' in options_dict.keys():
        gray_num = int(options_dict['-g'])
    if '--gray' in options_dict.keys():
        gray_num = int(options_dict['--gray'])

    if '-f' in options_dict.keys():
        fluorescence = True

    print(f"infile={infile}, outdir={outdir}, options={options_dict}")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # do something
    # 图片读取
    img = cv2.imread(infile)
    # 图片预处理
    if fluorescence:
        img_binary = fl_img_process(img)
    else:
        img_binary = he_img_process(img, gray_num)
    # show_img(img_binary)
    # HE区域识别
    area_sum, filter_conts = search_he_region(img_binary)
    # 将HE轮廓绘制在原图片中，并保存至指定目录
    img2 = img.copy()
    cv2.drawContours(img2, filter_conts, -1, (0, 0, 255), -1)
    cv2.imwrite(f'{outdir}/he_roi.tif', img2)
    # 保存小图片，用于网页版报告
    img2_small = cv2.resize(img2, (1024, int(img2.shape[0] / (img2.shape[1] / 1024))))
    cv2.imwrite(f'{outdir}/he_roi_small.png', img2_small)
    # 将HE区域轮廓保存进JSON格式文件中，用于后期使用
    save_roi_2_file(f'{outdir}/roi_heAuto.json', filter_conts)
    # 将图片大小信息与HE区域覆盖面积占比信息输出到文件中
    write_stat_2_file(f'{outdir}/stat.txt', img, area_sum)


def erode_dilate_process(img, process_info_list):
    '''
    腐蚀膨胀系列处理
    :param img:
    :param process_info_list: [['e', 11, 9], ['d', 11, 7],....] 表示先腐蚀，ksize=(11, 11), 迭代9次，然后膨胀...
    :return: img_dist
    '''
    for process in process_info_list:
        if process[0] == 'e':  # 腐蚀
            img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ERODE, (process[1], process[1])),
                             iterations=process[2], borderValue=0)
        if process[0] == 'd':  # 膨胀
            img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ERODE, (process[1], process[1])),
                             iterations=process[2], borderValue=0)
        show_img(img)
    return img


if __name__ == "__main__":
    # main()
    img = tifffile.imread(r"D:\code\cell-seg\img\img_dist_fqys.tif")
    filter_conts = get_img_conts(img, 'he')  # 找荧光图组织边界
    cv2.drawContours(img, filter_conts, -1, (0, 255, 0), 10)
    tifffile.imwrite("D:\code\cell-seg\img\itest.tif", cv2.resize(img, (5000, 5000)),
                     compression="jpeg")





