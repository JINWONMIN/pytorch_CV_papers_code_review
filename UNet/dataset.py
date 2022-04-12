import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더 구현
class Dataset(torch.utils.data.Dataset):    # Dataset 클래스에 torch.utils.data.Dataset 클래스를 상속
    def __init__(self, data_dir, transform=None):   # 할당 받을 인자 선언 (첫 선언)
        self.data_dir = data_dir
        self.transform = transform

        # prefixed word를 이용해 prefixed 되어 있는 input / label data를 나눠서 불러오기
        lst_data = os.listdir(self.data_dir) # os.listdir 메소드를 이용해서 data_dir에 있는 모든 파일을 불러온다.

        # prefixed 되어있는 word를 기반으로 label / input list를 정렬
        lst_label = [f for f in lst_data if f.startswith('label')]  # startswith 메소드를 통해
        lst_input = [f for f in lst_data if f.startswith('input')]  # prefixed 돼있는 리스트만 따로 정리.

        lst_label.sort()
        lst_label.sort()

        self.lst_label = lst_label  # 정렬된 리스트를
        self.lst_input = lst_input  # 해당 클래스의 파라미터로 가져온다.

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):   # index를 인자로 받아서 index에 해당하는 파일을 로드해서 리턴하는 형태로 정의
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) # NumPy 형식으로 저장되어 있어 np.load로 불러온다.
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0   # 데이터가 0 ~ 255 사이의 값으로 있기 때문에
        input = input / 255.0   # 이를 0 ~ 1 사이로 normalize

        if label.ndim == 2:     # input 값은 최소 3차원으로 넣어야 해서 2차원인 경우
            label = label[:, :, np.newaxis]     # 마지막 axis 임의로 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 생성된 label 과 input을 딕셔너리 형태로 저장

        if self.transform:  # transform function을 data loader의 argument로 넣어주고,
            data = self.transform(data)     # transform 함수가 정의 되어 있다면, transform 함수를 통과한 data를 리턴

        return data


## Data transform 구현하기
class ToTensor(object): # (NumPy -> Tensor)
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # [Image의 numpy 차원 = (Y, X, CH)] -> [Image의 pytorch tensor 차원 = (CH, Y, X)]
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}  # from_numpy 함수: numpy를 tensor로 넘겨줌.

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std
        # 여기서 label data는 0 or 1 클래스로 되어 있어서 정규화 X

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)    # 항상 input 과 label을 동시에 해줘야한다.
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data