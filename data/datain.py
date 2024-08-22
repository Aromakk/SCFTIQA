import os
import torch
import numpy as np
import cv2 


class DataIn(torch.utils.data.Dataset):
    def __init__(self, dis_path, sal_path, txt_file_name, transform):
        super(DataIn, self).__init__()
        self.dis_path = dis_path
        self.sal_path = sal_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                # print(dis)
                score = float(score)
                dis_files_data.append(dis)
                score_data.append(score)


        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        # png格式的sal图像
        d_sal_name = d_img_name.replace(".bmp",".png")

        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_sal = cv2.imread(os.path.join(self.sal_path, d_sal_name), cv2.IMREAD_GRAYSCALE)
        # print(type(d_sal))
        sal_h = d_sal.shape[0]
        sal_w = d_sal.shape[1]

        d_sal = d_sal.reshape(sal_h,sal_w,1)
        # print("success")

        # d_sal = d_sal.resize(500,500,1)
        # print(d_sal.shape)

        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_sal = np.array(d_sal).astype('float32') / 255

        d_img = np.transpose(d_img, (2, 0, 1))
        d_sal = np.transpose(d_sal, (2, 0, 1))


        
        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'd_sal_org': d_sal,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)



        return sample
