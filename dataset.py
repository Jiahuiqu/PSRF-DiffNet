import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import numpy as np


class Datasat(Dataset):
    def __init__(self, mode, size, IN_CH_HSI=31, IN_CH_MSI=3):

        super(Datasat, self).__init__()
        self.band = IN_CH_HSI
        self.size = int(size)
        self.img_path1 = []
        self.img_path2 = []
        self.img_path3 = []
        self.upSample = torch.nn.Upsample(scale_factor=4, mode='bicubic')

        path = './data/PaviaC/'
        N = 189
        if mode == 'train':
            self.GTHS_path = path+'/train/gtHS'
            self.img_path1 = os.listdir(self.GTHS_path)[0:N]
            sorted(self.img_path1, key=lambda x:int(x.split('.')[0]))
            self.HRMS_path = path+'/train/hrMS'
            self.img_path2 = os.listdir(self.HRMS_path)[0:N]
            sorted(self.img_path2, key=lambda x: int(x.split('.')[0])) ##_Elastic1000
            self.LRHS_path = path+'/train/LRHS_Elastic1000'
            self.img_path3 = os.listdir(self.LRHS_path)[0:N]
            sorted(self.img_path3, key=lambda x: int(x.split('.')[0]))
            print('训练数据初始化')
        if mode == 'test':
            self.GTHS_path = path+'/test/gtHS'
            self.img_path1 = os.listdir(self.GTHS_path)[0:N]
            self.img_path1.sort(key=lambda x: int(x.split(".")[0]))
            sorted(self.img_path1, key=lambda x: int(x.split('.')[0]))
            self.HRMS_path = path+'/test/hrMS'
            self.img_path2 = os.listdir(self.HRMS_path)[0:N]
            self.img_path2.sort(key=lambda x: int(x.split(".")[0]))
            sorted(self.img_path2, key=lambda x: int(x.split('.')[0]))
            self.LRHS_path = path+'/test/LRHS_Elastic1000'
            self.img_path3 = os.listdir(self.LRHS_path)[0:N]
            self.img_path3.sort(key=lambda x: int(x.split(".")[0]))
            sorted(self.img_path3, key=lambda x: int(x.split('.')[0]))
            print('测试数据初始化')

        self.gtHS = []
        self.hrMS = []
        self.lrHS = []
        self.HSI_Patch = []
        self.MSI_Patch = []
        self.MSI_Patch2 = []

        for i in range(len(self.img_path1)):
            MSI_Patch = np.zeros((IN_CH_MSI, size, size, 9))
            HSI_Patch = np.zeros((IN_CH_HSI, size, size, 9))
            MSI_Patch2 = np.zeros((IN_CH_MSI * 9, size, size, 9))
            pad_width = [(0, 0), (2, 2), (2, 2)]
            self.real_GTHS_path = os.path.join(self.GTHS_path, self.img_path1[i])
            gtHS_temp = loadmat(self.real_GTHS_path)['gtHS'].reshape(self.band, self.size, self.size)
            self.gtHS.append(gtHS_temp)
            self.real_HRMS_path = os.path.join(self.HRMS_path, self.img_path2[i])
            hrMS_temp = loadmat(self.real_HRMS_path)['hrMS'].reshape(4, self.size, self.size)
            self.hrMS.append(hrMS_temp)
            self.real_LRHS_path = os.path.join(self.LRHS_path, self.img_path3[i])
            lrhs_temp_org = loadmat(self.real_LRHS_path)['LRHS'].reshape(self.band, self.size // 4, self.size // 4)
            lrhs_temp = self.upSample(torch.from_numpy(lrhs_temp_org).unsqueeze(0)).squeeze(0).numpy()
            self.lrHS.append(lrhs_temp_org)

            padded_hrMS = np.pad(hrMS_temp, pad_width, mode='constant', constant_values=0)
            padded_lrHS = np.pad(lrhs_temp, pad_width, mode='constant', constant_values=0)

            HSI_Patch[:, :, :, 0] = padded_lrHS[:, 1:-3, 1:-3]
            HSI_Patch[:, :, :, 1] = padded_lrHS[:, 2:-2, 1:-3]
            HSI_Patch[:, :, :, 2] = padded_lrHS[:, 3:-1, 1:-3]
            HSI_Patch[:, :, :, 3] = padded_lrHS[:, 1:-3, 2:-2]
            HSI_Patch[:, :, :, 4] = padded_lrHS[:, 2:-2, 2:-2]
            HSI_Patch[:, :, :, 5] = padded_lrHS[:, 3:-1, 2:-2]
            HSI_Patch[:, :, :, 6] = padded_lrHS[:, 1:-3, 3:-1]
            HSI_Patch[:, :, :, 7] = padded_lrHS[:, 2:-2, 3:-1]
            HSI_Patch[:, :, :, 8] = padded_lrHS[:, 3:-1, 3:-1]
            self.HSI_Patch.append(HSI_Patch.transpose(0, 3, 1, 2).reshape(-1, size, size))

            MSI_Patch[:, :, :, 0] = padded_hrMS[:, 1:-3, 1:-3]
            MSI_Patch[:, :, :, 1] = padded_hrMS[:, 2:-2, 1:-3]
            MSI_Patch[:, :, :, 2] = padded_hrMS[:, 3:-1, 1:-3]
            MSI_Patch[:, :, :, 3] = padded_hrMS[:, 1:-3, 2:-2]
            MSI_Patch[:, :, :, 4] = padded_hrMS[:, 2:-2, 2:-2]
            MSI_Patch[:, :, :, 5] = padded_hrMS[:, 3:-1, 2:-2]
            MSI_Patch[:, :, :, 6] = padded_hrMS[:, 1:-3, 3:-1]
            MSI_Patch[:, :, :, 7] = padded_hrMS[:, 2:-2, 3:-1]
            MSI_Patch[:, :, :, 8] = padded_hrMS[:, 3:-1, 3:-1]
            MSI_Patch = MSI_Patch.transpose(0, 3, 1, 2).reshape(-1, size, size)
            self.MSI_Patch.append(MSI_Patch.reshape(-1, size*size))
            padded_MSI_Patch = np.pad(MSI_Patch, pad_width, mode='constant', constant_values=0)

            MSI_Patch2[:, :, :, 0] = padded_MSI_Patch[:, 1:-3, 1:-3]
            MSI_Patch2[:, :, :, 1] = padded_MSI_Patch[:, 2:-2, 1:-3]
            MSI_Patch2[:, :, :, 2] = padded_MSI_Patch[:, 3:-1, 1:-3]
            MSI_Patch2[:, :, :, 3] = padded_MSI_Patch[:, 1:-3, 2:-2]
            MSI_Patch2[:, :, :, 4] = padded_MSI_Patch[:, 2:-2, 2:-2]
            MSI_Patch2[:, :, :, 5] = padded_MSI_Patch[:, 3:-1, 2:-2]
            MSI_Patch2[:, :, :, 6] = padded_MSI_Patch[:, 1:-3, 3:-1]
            MSI_Patch2[:, :, :, 7] = padded_MSI_Patch[:, 2:-2, 3:-1]
            MSI_Patch2[:, :, :, 8] = padded_MSI_Patch[:, 3:-1, 3:-1]
            MSI_Patch2 = MSI_Patch2.reshape(-1, size*size, 9)
            self.MSI_Patch2.append(MSI_Patch2)
        print('数据初始化完成')

    def __getitem__(self, item):

        gtHS = self.gtHS[item]
        hrMS = self.hrMS[item]
        lrHS = self.lrHS[item]
        HSI_Patch = self.HSI_Patch[item]
        MSI_Patch2 = self.MSI_Patch2[item]

        return gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2

    def __len__(self):
        return len(self.img_path1)

if __name__ == '__main__':
    pass

