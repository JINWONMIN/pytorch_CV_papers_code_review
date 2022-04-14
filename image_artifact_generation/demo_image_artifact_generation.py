## import packages
import numpy as np
from scipy.stats import poisson
from scipy.io import loadmat

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale

import matplotlib.pyplot as plt

## Load image
img = plt.imread("lenna.png")

# gray image generation --> gray image로 만들기 위해선 채널 방향으로 평균을 내주면 된다.
# img = np.mean(img, axis=2, keepdims=True)   # axis=2 -> 3번째 (W, H, C) === 즉 채널 방향

sz = img.shape

cmap = "gray" if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1 )
plt.title("Ground Truth")
plt.show()

'''
1. Sampling mask 생성하기
a) uniform mask
b) uniform random mask
c) gaussian random mask
'''

## 1-1. Inpainting: Uniform sampling

# define x direction and sampling ratio
ds_y = 2    # interval of y
ds_x = 4    # interval of x

msk = np.zeros(sz)  # create mask
msk[::ds_y, ::ds_x, :] = 1  # sampling at interval ds_y and ds_x

dst = img*msk   # mak의 간격으로 sampling하기 위해 곱해준다.

'''
시각화
subplot(131) - Ground Truth 
subplot(132) - Uniform sampling mask: y 방향으로 샘플링 ratio = 2, x 방향으로 샘플링 ratio = 4인 샘플링 마스크
                ex) y축 방향은 [0번 idx = sampling, 1번 idx = not sampling, ...]
                    x축 방향은 [0번 idx = sampling, 1번 idx = not sampling, 2번 idx = not sampling, 3번 idx = not sampling, ...] 
subplot(133) - Sampling image: 샘플링 마스크가 적용된 샘플링 된 이미지를 보여준다.
'''
plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

## 1-2. Inpainting: random sampling
'''
mask를 만들고 각 이미지마다 mask를 이미지에 곱하는 형태로 random sampling 진행
'''
# rnd = np.random.rand(sz[0], sz[1], sz[2])   # 1) random sampling을 하기 위해 uniform size와 동일안 random variables 생성
# prob = 0.5  # sampling 할 비율
# msk = (rnd > prob).astype(np.float)

rnd = np.random.rand(sz[0], sz[1], 1) # 채널 방향으로 복사해주면 된다. (채널 방향으로는 동일한 마스크가 진행된다.)
prob = 0.5
msk = (rnd > prob).astype(np.float)

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Random mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling Image")

## 1-3. Inpainting: Gaussian sampling: 분포는 Gaussian distribution를 따르지만 센터에 샘플링 포인트들이 많고 센터에서 멀어질수록 샘플링 포인트의 수가 적어지는 마스크.

# define 2d Gaussian distribution function
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0  # center를 기준
y0 = 0  # center를 기준
sgmx = 1
sgmy = 1

a = 1

# gaus = a * np.exp(- ((x - x0 )**2 / (2 * sgmx**2) + (y - y0)**2 / (2 * sgmy**2)))   # 2d gaussian distribution 공식
# plt.imshow(gaus)
# gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gaus).astype(np.float)

gaus = a * np.exp(- ((x - x0 )**2 / (2 * sgmx**2) + (y - y0)**2 / (2 * sgmy**2)))
gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))

rnd = np.random.rand(sz[0], sz[1], 1)
msk = ( rnd < gaus).astype(np.float)
msk = np.tile(msk, (1, 1, sz[2]))

dst = img * msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling Image")


'''
2. Noise 생성하기 (denoising task를 할 때 주로 사용)
a) normal random noise
b) poisson random noise

bm3d 참고: \sigma에 따라서 해당 이미지에 노이즈가 달라진다. 값이 커질수록 노이즈의 양이 증가한다.
'''
## 2-1. Denoising: Random nosie
sgm = 30.0

noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])  # 이미지가 0~1 사이로 normalization이 되어 있어서 시그마에도 normalization을 해줘서 스케일을 맞춰준다.

dst = img + noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title("Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Noisy Image")

## 2-2. Denoising: poisson noise (image-domain) <포아송 노이즈> : 영상 복원할 때 주로 사용됨. (ex) ct촬영
dst = poisson.rvs(255.0 * img) / 255.0  # https://hyperpolyglot.org/numerical-analysis2 (MATLAB 과 python가 동일한 함수가 정리되어 있음.)
noise = dst - img

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title("Poisson Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Noisy Image")

## 2-3. Denoising: poisson noise (CT-domain)
# SYSTEM SETTING
N = 512
ANG = 180
VIEW = 360
THETA = np.linspace(0, ANG, VIEW, endpoint=False)

A = lambda  x: radon(x, THETA, circle=False).astype(np.float32)
AT = lambda  y: iradon(y, THETA, circle=False, filter=None, output_size=N).astype(np.float32)
AINV = lambda y: iradon(y, THETA, circle=False, output_size=N).astype(np.float32)

# Low dose CT: adding poisson noise
pht = shepp_logan_phantom()
pht = 0.03 * rescale(pht, scale=512/400, oreder=0)

prj = A(pht)

i0 = 1e4
dst = 10 * np.exp(-prj)
dst = poisson.rvs(dst)
dst = -np.log(dst / i0)
dst[dst < 0] = 0

noise = dst - prj

rec = AINV(prj)
rec_noise = AINV(noise)
rec_dst = AINV(dst)


## 3. Super-resolution: downsampling된 이미지를 원본 이미지로 복원하는 방법
'''
-------------------------
order options
-------------------------
0: Nearest-neighbor
1: Bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic
'''
# down sampling / up sampling
dw = 1 / 5.0    # downsampling ratio 5배
order = 0   # Nearest-neighbor

dst_dw = rescale(img, scale=(dw, dw, 1), order=order)    # scale(y, x, ch)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1), order=order)

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(dst_dw), cmap=cmap, vmin=0, vmax=1)
plt.title("Downscaled image")

plt.subplot(133)
plt.imshow(np.squeeze(dst_up), cmap=cmap, vmin=0, vmax=1)
plt.title("Unscaled image")
























