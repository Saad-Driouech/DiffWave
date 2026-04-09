import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pickle
import math

def cossin_to_angle_deg(cossin):
    """Convert sin/cos representation to angle in degrees"""
    if type(cossin) == list:
        cossin = torch.tensor(cossin)
    if type(cossin) == np.ndarray:
        return np.arctan2(cossin[:, 0], cossin[:, 1]) * 180.0 / math.pi
    return torch.atan2(cossin[:, 0], cossin[:, 1]) * 180.0 / math.pi

def angle_deg_to_cossin(angle):
    """Convert angle in degrees to sin/cos representation"""
    if type(angle) == list:
        angle = torch.tensor(angle)
    angle_rad = angle * math.pi / 180.0
    if type(angle) == np.ndarray:
        return np.stack([np.sin(angle_rad), np.cos(angle_rad)], axis=1)
    if isinstance(angle, (float, np.floating)):
        return np.array([np.sin(angle_rad), np.cos(angle_rad)])

    return torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)], dim=1)
class StationaryEttus(Dataset):
    def __init__(self, mode = "train"):
        self.mean = [4.8375e-08-6.3028e-07j,  1.3440e-08-7.7196e-07j, -2.4395e-07-1.7472e-07j, -1.7156e-07-1.3585e-07j, 6.9422e-08-2.8931e-07j, -2.4767e-07-2.2652e-07j, -1.9374e-07+9.2003e-08j, -1.1901e-07+7.1497e-07j]
        self.std = [0.0011, 0.0010, 0.0010, 0.0009, 0.0009, 0.0009, 0.0010, 0.0010]
        self.phase_offsets_start = [0., 4.20611423, 1.88940871, 2.12964778, 6.00147346, 2.22917107, 3.80224375, 1.56729763]
        if mode == "train":
            self.PATH_LBL = "/data/beegfs/darcy/Link_Halle_Ettus/Data_Recording/full_files/Blue_8_Antenna_Linear_Stationary_train_h5.txt"
        elif mode == "test":
            self.PATH_LBL = "/data/beegfs/darcy/Link_Halle_Ettus/Data_Recording/full_files/Blue_8_Antenna_Linear_Stationary_test_h5.txt"
        elif mode == "all":
            self.PATH_LBL = "/data/beegfs/darcy/Link_Halle_Ettus/Data_Recording/full_files/Blue_8_Antenna_Linear_Stationary_h5.txt"
        else:
            raise ValueError("Mode must be 'train', 'test' or 'all'")
        self.PATH_IQ = "/data/beegfs/darcy/Link_Halle_Ettus/Data_Recording/Blue_8_Antenna_Linear_Stationary.h5"
        with open("/data/beegfs/darcy/Link_Halle_Ettus/Data_Recording/full_files/Blue_8_Antenna_Linear_Stationary_phase_correction.pkl", "rb") as f:
            self.phase_corrections = pickle.load(f)
        self.h5_file  = None
        with open(self.PATH_LBL, "r") as f:
            data_info = f.readlines()
        data_info = np.array([x.split("\n")[0].split("\t") for x in data_info])
        files = []
        labels = []
        for x in data_info:
            files.append([x[0], int(x[1])])
            cur_y_labs = []
            cur_y_labs.append(float(x[int(8)]))
            cur_y_labs.append(float(x[int(9)]))
            cur_y_labs.append(float(x[int(10)]))
            cur_y_labs.append(float(x[int(11)]))
            cur_y_labs.append(float(x[int(12)]))
            labels.append(cur_y_labs)
        self.labels = np.array(labels)
        self.files = files
        self.x_data = []
        if self.h5_file is None:
            self.h5_file = h5py.File(self.PATH_IQ, 'r')

    def __getitem__(self, item: int):
        cur_x_data = self.h5_file[self.files[item][0]][self.files[item][1]][:, 500:]
        random_position = int(np.random.random()*(cur_x_data.shape[1] - 1024))
        cur_x_data = cur_x_data[:, random_position:random_position+1024] * np.exp(-1j * self.phase_corrections[self.files[item][0]])[:, None]
        cur_x_data = cur_x_data * np.exp(-1j * np.array(self.phase_offsets_start)[:, None])
        cur_x_data = (cur_x_data - np.array(self.mean)[:, None]) / np.array(self.std)[:, None]
        y1 = torch.tensor((self.labels[item][0], self.labels[item][1], self.labels[item][2]), dtype=torch.float)
        az_angle = angle_deg_to_cossin(self.labels[item][3])
        el_angle = angle_deg_to_cossin(self.labels[item][4])
        return torch.from_numpy(cur_x_data), (y1, az_angle, el_angle)

    def __len__(self) -> int:
        return len(self.labels[:, 0])