import argparse

import numpy as np
import random


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="计算百创S芯片系列重复率")

    parser.add_argument('-M', type=int, help="Total type num", required=True)
    parser.add_argument('-N', type=int, help="Pick num", required=True)
    args = parser.parse_args()

    # N = 55000000
    # M = 11000000
    M = args.M
    N = args.N

    # 生成一个包含10000个介于1到3之间的随机整数的数组
    random_array = np.array([random.randint(1, N) for x in range(M)])

    # 找到唯一值和它们的出现次数
    unique_values, counts = np.unique(random_array, return_counts=True)

    # 获取排序后的索引
    sorted_indices = np.argsort(counts)[::-1]

    # 根据排序后的索引获取排序后的唯一值和计数
    sorted_unique_values = unique_values[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # 打印排序后的结果
    # print("排序后的唯一值:", sorted_unique_values)
    # print("对应的计数:", sorted_counts[:10])

    # 下面计算重复的数目
    rep_value, rep_count = np.unique(sorted_counts, return_counts=True)
    rep_num = rep_value * rep_count
    rep_rate = rep_num / M
    rep_rate = rep_rate.astype(float)

    print("种类N：", N)
    print("取样数M：", M)
    print("\t".join(["重复", "数目", "比率"]))
    for val, num, rate in zip(rep_value, rep_num, rep_rate):
        print("\t".join([str(val), str(num), str(rate)]))