import numpy as np

from UMICountDistPlot import *
import cv2



supspot_pos_file = r'E:\biomarker_data\HE_UMI_Analyze\BSTViewer_project\level_matrix\level_1/barcodes_pos.tsv.gz'
matrix_file = r'E:\biomarker_data\HE_UMI_Analyze\BSTViewer_project\level_matrix\level_1/matrix.mtx.gz'
he_path = r"E:\01-key2_chip\img_dist.tif"
umi_path = r"E:\01-key2_chip\img_dist-umi.tif"


he_img = cv2.imread(he_path)
umi_img = np.zeros(he_img.shape, dtype=np.uint8)
zoom_scale = cal_zoom_rate(umi_img.shape[1], umi_img.shape[0])
umi_pos_df, umi_num = umi_stat(zoom_scale, supspot_pos_file, matrix_file)
# umi_plot(umi_img, umi_pos_df, umi_num, umi_path)

cn = plt.cm.get_cmap('binary')
# cn_colors = np.array(cn(umi_pos_df['umi'])[:, [2, 1, 0]] * 255, dtype='int')
cn_colors = np.array(cn(umi_pos_df['umi'])[:, [0, 1, 2]] * 255, dtype='int')
# 绘图-绘制每个spot
# radius = 30
radius = auto_cal_radius(umi_pos_df)
for index, item in umi_pos_df.iterrows():
    cv2.circle(umi_img, (round(item['pos_w']), round(item['pos_h'])), radius, list(map(int, cn_colors[index, :])),
               -1)
cv2.imwrite(umi_path, umi_img)