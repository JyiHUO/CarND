import os
import argparse
import pandas as pd

def main(args):
    dir_img1 = args.dir_img1
    dir_img2 = args.dir_img2
    csv_file1 = args.csv_file1
    csv_file2 = args.csv_file2
    # command = "for f in $(ls %s|grep jpg); do cp %s/$f %s;done"%(dir_img1, dir_img1, dir_img2)
    # print (command)
    # os.system(command)
    csv1 = pd.read_csv(csv_file1, header=None)
    csv2 = pd.read_csv(csv_file2, header=None)
    csv = pd.concat([csv1, csv2], axis=0)
    csv.to_csv(csv_file2, index=False, header=None)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dir_img1', type=str)
    args.add_argument('--dir_img2', type=str)
    args.add_argument('--csv_file1', type=str)
    args.add_argument('--csv_file2', type=str)
    parser = args.parse_args()
    main(parser)
