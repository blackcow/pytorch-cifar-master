'''Train CIFAR10 with PyTorch
load 模型 test 指标，进行 T-SNE 可视化
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import cifar10my2
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from time import time
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--imlabel', default=5, type=int, help='Label of the remove part of training data')
parser.add_argument('--dele', default=3, type=int, help='Label of the deleted training data')
parser.add_argument('--percent', default=0.1, type=float, help='Percentage of deleted data')
parser.add_argument('--gpu', default='1,2', type=str, help='GPUs id')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)

# 设定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 改写后的 load data
trainset = cifar10my2.CIFAR10MY(
    root='./data', train=True, download=True, transform=transform_train, args=args)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
    # trainset, batch_size=128, shuffle=False, num_workers=2)

bs = 1000
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = resnet.ResNet18()
net = torch.nn.DataParallel(net).cuda()
# net = net.cuda()

# ckpt = 'ckpt-alldata.pth'
# ckpt = 'ckpt_rmlabel1_percent0.2.pth'
# ckpt = 'ckpt_rmlabel1_percent0.9.pth'

# ckpt = 'ckpt_rmlabel0_percent0.2.pth'
ckpt = 'ckpt_rmlabel0_percent0.9.pth'

# label 3，5 受影响大
# ckpt = 'ckpt_rmlabel5_percent0.2.pth'
# ckpt = 'ckpt_rmlabel5_percent0.9.pth'

# ckpt = 'ckpt_rmlabel3_percent0.2.pth'
# ckpt = 'ckpt_rmlabel3_percent0.9.pth'
path = '/hot-data/niuzh/Mycode/pytorch-cifar-master/checkpoint/'
ckp_path = path + ckpt

checkpoint = torch.load(ckp_path)
# checkpoint = torch.load(ckp_path,map_location='cuda:1')
net.load_state_dict(checkpoint['net'])
# net.load_state_dict(cp1)
net.eval()

cudnn.benchmark = True

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 # color=plt.cm.Set1(label[i] / 10.),
                 color=plt.cm.Set3(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig

def test():
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out4, outputs = net(inputs)
            out4 = out4.reshape(bs, -1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # T-sne 可视化
            # tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=100, init='pca', random_state=0)
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time()
            out4 = out4.cpu().numpy()
            targets = targets.cpu().numpy()

            result = tsne.fit_transform(out4)
            fig = plot_embedding(result, targets, 't-SNE embedding of the CIFAR-10 (time %.2fs)' % (time() - t0))
            aa = 0
    # Save checkpoint.
    acc = 100.*correct/total
    return best_acc, best_epoch

best_acc, best_epoch = test()


print("model best acc：%f ,epoch %d" % (best_acc, best_epoch))
print(args)