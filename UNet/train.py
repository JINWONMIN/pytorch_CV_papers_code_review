## import libraries
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## Parser 생성
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode") # train or test를 구분하기 위한 파서
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## training paremeters setting
lr = args.lr    # 1e-3
batch_size = args.batch_size    # 4
num_epoch = args.num_epoch  # 100

# local directory path
data_dir = args.data_dir  # './datasets'
ckpt_dir = args.ckpt_dir  # './checkpoint'   # trained data save directory
log_dir = args.log_dir    # './log'           # 텐서보드 로그가 저장될 디렉토리
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

# colab directory path
# data_dir = './drive/MyDrive/pytorch/UNet_36/datasets'
# ckpt_dir = './drive/MyDrive/pytorch/UNet_36/checkpoint'
# log_dir = './drive/MyDrive/pytorch/UNet_36/log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
# 학습시킬 데이터를 불러오기 위해 transform 함수 정의
# if __name__ == "__main__":
if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()]) # transforms.Compose: 여러 개의 transform 함수들을 묶어서 사용할 수 있다.

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)  # test set 불러오기
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)  # val set 불러오기
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    ## 그 밖의 부수적인 variables 설정
    # training / val set의 갯수를 설정하는 변수 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])  # transforms.Compose: 여러 개의 transform 함수들을 묶어서 사용할 수 있다.

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)  # test set 불러오기
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖의 부수적인 functions 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # tonumpy: from tensor to numpy
fn_denorm = lambda x, mean, std: (x * std) + mean   # denormalization()
fn_class = lambda  x: 1.0 * (x > 0.5)   # classification() using thresholding (p=0.5) <네트워크 output 이미지를 binary class로 분류해주는 function

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0    # 트레이닝이 시작되는 epoch의 position을 0으로 설정

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train() # network에게 training mode임을 알려주는 train() 활성화
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward path
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산

            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))


            # Tensorboard 저장 (label, input, output)
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch -1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)   # loss를 텐서보드에 저장
        # network validation
        with torch.no_grad(): # validation 부붑은 backpropergation하는 부분이 없기때문에 backpropergation을 막기 위해 torch.no_grad() 을 활성화.
            net.eval()  # network에게 validation mode임을 알려주는 train() 활성화
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장 (label, input, output)
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)  # loss를 텐서보드에 저장

            if epoch % 50 == 0: # (n(50)번 마다 저장하고 싶을 때 사용)
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)  # epoch가 한 번 진행될때마다 네트워크 저장

        writer_train.close() # 학습이 완료되면 tensorboard를 저장하기 위해 생성했던 두 개의 writer를 close()
        writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():  # validation 부붑은 backpropergation하는 부분이 없기때문에 backpropergation을 막기 위해 torch.no_grad() 을 활성화.
        net.eval()  # network에게 validation mode임을 알려주는 eval() 활성화
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward path
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장 (label, input, output)
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                # save as png type
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # save as numpy type
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))














