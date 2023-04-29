import os
import torch
import numpy as np
import tqdm.auto as tqdm
import imageio.v3 as iio
import matplotlib.pyplot as plt

#@title Unlabeled Images

P = "/dataset"

unlabeled_imgs = torch.zeros([13000,22,160,240,3])
for path in tqdm.tqdm(os.listdir(f"{P}/unlabeled/")):
    temp = []
    for i in range(22):
        copy = np.copy(torch.Tensor(iio.imread(f"{P}/unlabeled/{path}/image_{i}.png")))
        temp.append(torch.Tensor(copy))
    unlabeled_imgs[int(path.split("_")[1])-2000] = torch.stack(temp)
torch.save(unlabeled_imgs, 'unlabeled_imgs.pt')
