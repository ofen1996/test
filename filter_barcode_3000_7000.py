import os


if __name__ == '__main__':
    base_dir = r"./"

    txt_names = ["bc_1.txt", "bc_2.txt", "bc_3.txt"]
    for name in txt_names:
        # name = txt_names[0]
        with open(os.path.join(base_dir, name), "r") as fp:
            all_barcode = fp.read()
        all_barcode = [_.split("\t") for _ in all_barcode.split("\n") if _ != '']
        filter_barcode = list(filter(lambda x: not(3000 < int(x[2]) < 7000), all_barcode))
        with open(os.path.join(base_dir, "filter_"+name), "w") as fp:
            fp.write("\n".join(["\t".join(x) for x in filter_barcode]))