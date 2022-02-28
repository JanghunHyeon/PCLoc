import numpy as np
import os
import glob
import pandas as pd

def load_transformation(pth):
    """
    pth : ./wust1_dataset/database/alignments/CSE3/...
    """

    P_before = np.zeros((4, 4))
    P_after = np.zeros((4, 4))

    fid = pd.read_csv(pth, header=None, sep='\n')
    data_all = fid[0]

    P_before[0, :] = np.double(data_all[1].split())
    P_before[1, :] = np.double(data_all[2].split())
    P_before[2, :] = np.double(data_all[3].split())
    P_before[3, :] = np.double(data_all[4].split())

    P_after[0, :] = np.double(data_all[6].split())
    P_after[1, :] = np.double(data_all[7].split())
    P_after[2, :] = np.double(data_all[8].split())
    P_after[3, :] = np.double(data_all[9].split())

    return P_before, P_after


if __name__ == '__main__':

    pth = '/mnt/hdd2/Dataset/InLoc_sample/database/alignments/CSE3/transformations/cse_trans_000.txt'
    load_transformation(pth)