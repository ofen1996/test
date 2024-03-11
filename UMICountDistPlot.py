import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import getopt
import sys
import os
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# 依据HE图片的实际大小进行缩放比例计算
# width：HE图片的宽
# height: HE图片的高
def cal_zoom_rate(width, height):
    std_width = 1000
    std_height = std_width / (46 * 31) * (46 * 36 * np.sqrt(3) / 2.0)
    if std_width / std_height > width / height:
        scale = width / std_width
    else:
        scale = height / std_height
    return scale

# 自动进行半径计算
def auto_cal_radius(cluster_pos_df):
    radius = 999999
    pref_pos = [0,0]
    for index, item in cluster_pos_df.iterrows():
        if index != 0:
            curr_pos = [item['pos_w'], item['pos_h']]
            center_dist = np.sqrt((curr_pos[0]-pref_pos[0])**2 + (curr_pos[1]-pref_pos[1])**2)
            if center_dist < radius:
                radius = center_dist
        pref_pos = [item['pos_w'], item['pos_h']]
        if index > 1000:
            break
    radius = round(radius * 0.618 / 2)
    if radius < 1:
        radius = 1
    return radius


def umi_stat(zoom_scale, supspot_pos_file, matrix_file):
    # 读取supspot点位置信息
    pos_df = pd.read_csv(supspot_pos_file, compression='gzip', sep='\t', names=['name', 'pos_w', 'pos_h'], header=None)
    # 依据图片缩放率，对坐标进行，并进行取整数
    pos_df['pos_w'] = pos_df['pos_w'] * zoom_scale
    pos_df['pos_h'] = pos_df['pos_h'] * zoom_scale
    pos_df.round({'pos_w': 0, 'pos_h': 0})
    pos_df['supspot'] = range(1, 1 + pos_df.shape[0])

    # 从文件中读取表达量矩阵
    df = pd.read_csv(matrix_file, compression='gzip', sep='\t', names=['gene', 'supspot', 'umi'], skiprows=3,
                     header=None)
    # 依据supspot进行分类对umi count求和
    umi_num = df.groupby('supspot', as_index=False).agg({'umi': 'sum'}).copy()
    #umi_num['supspot'] = umi_num['supspot'] - 1
    # 依据supspot将umi count值与位置坐标进行合并，得到一个新的dataframe
    umi_pos_df = pd.merge(pos_df, umi_num, how='left')
    umi_pos_df.fillna(0, inplace=True)
    umi_pos_df['umi'] = umi_pos_df['umi'] / umi_pos_df['umi'].max()
    print(f"max umi count:{umi_num['umi'].max()}, min umi count:{umi_num['umi'].min()}")

    return umi_pos_df,umi_num
    pass



# 绘制color bar
def draw_color_bar(img, max_value):
    # 定义尺寸
    img_w,img_h = img.shape[0:2]
    bar_pos = [0,0]
    bar_h = int(img_h/3)
    bar_w = int(img_w/50)
    bar_scale_h = max(int(bar_h/200),1)
    bar_scale_w = int(bar_w/3)
    # 绘制color bor
    # vals = range(0, bar_h,1)
    vals = np.arange(0,bar_h,1)
    cn = plt.cm.get_cmap('RdYlBu_r')
    cn_colors = np.array(cn(vals/(bar_h-1))[:,[2,1,0]]*255,dtype='uint8')
    for h_i in range(bar_h):
        for w_i in range(bar_w):
            img[bar_pos[0]+h_i,bar_pos[1]+w_i,:] = cn_colors[bar_h-1-h_i,:]
    # 绘制刻度
    num_len = len(str(max_value))
    scale_mod_dict = [1, 2, 3, 5, 5, 10, 10, 10, 10, 10]
    scale_mod = scale_mod_dict[int(str(max_value)[0])] * np.power(10, num_len - 2)
    scale_val = np.arange(scale_mod,max_value,scale_mod)
    scale_pos = np.array(scale_val / max_value * bar_h, dtype=int)
    scale_pos = bar_pos[0] + bar_h - scale_pos - int(bar_scale_h/2)
    print(f"val:{scale_val}, pos:{scale_pos}")
    pos_w = bar_pos[1] + bar_w
    print(f"w:{bar_scale_w},pos_w:{pos_w}")
    for pos_h in scale_pos:
        for h_i in range(pos_h - int(bar_scale_h / 2), pos_h - int(bar_scale_h / 2) + bar_scale_h, 1):
            for w_i in range(bar_scale_w):
                img[h_i, pos_w+w_i, :] = [0,0,0]
    # 绘制文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos_w = bar_pos[1] + bar_w + bar_scale_w + int(bar_scale_w/2)
    for i in range(len(scale_pos)):
        textSize, baseline = cv2.getTextSize(str(scale_val[i]), font, 2.0, 2)
        print(f"size:{textSize}, baseline:{baseline}")
        pos_h = scale_pos[i] - int(bar_scale_h / 2) + int(textSize[1]/2)
        cv2.putText(img, str(scale_val[i]), (pos_w,pos_h),font, 2.0, (0,0,0),2)

    # he_img = cv2.putText(he_img, '000', (50, 300), font, 1.2, (255, 255, 255), 2)
    pass


# 计算刻度
def get_color_bar_tick_val(max_value):
    num_len = len(str(max_value))
    scale_mod_dict = [1, 2, 3, 5, 5, 10, 10, 10, 10, 10]
    scale_mod = scale_mod_dict[int(str(max_value)[0])] * np.power(10, num_len - 2)
    scale_val = np.arange(scale_mod,max_value,scale_mod)
    scale_pos = np.array(scale_val / max_value, dtype=float)
    print(f"val:{scale_val}, pos:{scale_pos}")

    return scale_val,scale_pos


# 画图
def umi_plot(he_img, umi_pos_df, umi_num, outfile):
    # 依据umi count值生成颜色值
    cn = plt.cm.get_cmap('RdYlBu_r')
    # cn_colors = np.array(cn(umi_pos_df['umi'])[:, [2, 1, 0]] * 255, dtype='int')
    cn_colors = np.array(cn(umi_pos_df['umi'])[:, [0, 1, 2]] * 255, dtype='int')
    # 绘图-绘制每个spot
    # radius = 30
    radius = auto_cal_radius(umi_pos_df)
    for index, item in umi_pos_df.iterrows():
        cv2.circle(he_img, (round(item['pos_w']), round(item['pos_h'])), radius, list(map(int, cn_colors[index, :])),
                   -1)
    # 用plt绘图color bar
    scale_val,scale_pos = get_color_bar_tick_val(umi_num['umi'].max())
    plt.figure(figsize=(6, 6))
    plt.imshow(he_img)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.get_cmap('RdYlBu_r', 50)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    Figlegend = plt.colorbar(im, shrink=0.75, ax=plt.gca())
    colorbarfontdict = {"size": 8, "color": "k", 'family': 'Times New Roman'}
    legend_title = 'nUMI'
    Figlegend.locator = mpl.ticker.FixedLocator(scale_pos)
    Figlegend.formatter = mpl.ticker.FixedFormatter(scale_val)
    Figlegend.update_ticks()

    Figlegend.ax.set_title(legend_title, fontdict=colorbarfontdict, pad=4)
    Figlegend.ax.tick_params(labelsize=6, direction='out')

    # 保存图片
    plt.axis('off')  # 去掉边框
    plt.tight_layout()
    plt.savefig(f"{outfile}", dpi=300)
    plt.savefig(f"{outfile[:-4]}_small.png", dpi=150)
    pass


def do_something(used_args):
    # 读取HE图片
    he_img = cv2.imread(used_args['he_file'])
    # 依据HE图片尺寸信息计算图片缩放率
    zoom_scale = cal_zoom_rate(he_img.shape[1], he_img.shape[0])

    # umi统计
    supspot_pos_file = f'{used_args["indir"]}/barcodes_pos.tsv.gz'
    # supspot_pos_file = 'D:/delete/smaSTViewer/project_JM1_25G/subdata/L13_Lall/barcodes_pos.tsv.gz'
    # 表达量矩阵文件
    matrix_file = f'{used_args["indir"]}/matrix.mtx.gz'
    umi_pos_df,umi_num = umi_stat(zoom_scale, supspot_pos_file, matrix_file)

    # plot
    umi_plot(he_img, umi_pos_df, umi_num, used_args['outfile'])
    pass


def usage():
    print("======="*10)
    print("Usage:")
    print("1) python UMICountDistPlot.py -e ./test.tif -i ./outdir/ -o umi_dist.tif")
    print("2) python UMICountDistPlot.py --he_file ./test.tif --indir ./outdir/ --outfile umi_dist.tif")
    print("3) python UMICountDistPlot.py -h")
    print("4) python UMICountDistPlot.py --help")
    print("======="*10)
    exit()



def main():
    # init
    short_opts = 'he:i:o:'
    long_opts = ['help', 'he_file=', 'indir=', 'outfile=']
    args_names = [None, 'he_file', 'indir', 'outfile']

    # 生成required_options与opts_2_names
    ss = short_opts.replace(':','')
    required_options = []
    opts_2_names = {}
    for i in range(len(args_names)):
        if args_names[i] is None:
            continue
        else:
            short_o = '-' + ss[i]
            long_o = '--' + long_opts[i].replace('=','')
            required_options.append([short_o,long_o])
            opts_2_names[short_o] = args_names[i]
            opts_2_names[long_o] = args_names[i]

    # getopt
    options_dict = {}
    used_args = {}
    options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
    for name, value in options:
        if name not in options_dict.keys():
            options_dict[name] = value
        if name in opts_2_names.keys():
            used_args[opts_2_names[name]] = value
    print(used_args)

    if '-h' in options_dict.keys() or '--help' in options_dict.keys():
        usage()

    # 检查必需提供的参数是否被提供，没有就提示一下，并打印使用说明
    for short_o, long_o in required_options:
        if short_o not in options_dict.keys() and long_o not in options_dict.keys():
            print(f"{short_o} or {long_o} must exist")
            usage()

    do_something(used_args)



if __name__ == "__main__":
    main()





