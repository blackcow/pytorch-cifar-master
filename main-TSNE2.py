'''Train CIFAR10 with PyTorch
load 模型 test 指标，进行 T-SNE 可视化
TRADES 自己训练的模型
需要手动调整 epsilon，更换模型和 adv example
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
from models.wideresnet import WideResNet
import random
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--imlabel', default=5, type=int, help='Label of the remove part of training data')
parser.add_argument('--dele', default=3, type=int, help='Label of the deleted training data')
parser.add_argument('--percent', default=0.1, type=float, help='Percentage of deleted data')
parser.add_argument('--gpu', default='0,1,2', type=str, help='GPUs id')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epsilon', default=0.031, type=float, help='perturbation')
parser.add_argument('--num-steps', default=20, help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, help='perturb step size')
parser.add_argument('--random', default=True, help='random initialization for PGD')
args = parser.parse_args()
print(args)

# 设定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

seed=0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False


# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # 对于 TRADES 注释掉
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 改写后的 load data
# trainset = cifar10my2.CIFAR10MY(
#     root='./data', train=True, download=True, transform=transform_train, args=args)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)
#     # trainset, batch_size=128, shuffle=False, num_workers=2)

# bs = 1000
bs = 50
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
factors = [0.031, 0.0156, 0.0078, 0.0039]
epsilon = factors[3]
model_name = 'e' + str(epsilon) +'_depth34_widen10_drop0.0'
# model_name = 'e0.0156_depth34_widen10_drop0.0'
# model_name = 'e0.0078_depth34_widen10_drop0.0'
# model_name = 'e0.0039_depth34_widen10_drop0.0'
ckpt = '/hot-data/niuzh/Mycode/TRADES-master/model-cifar-wideResNet/AT/' + model_name +'/'
ckpt += 'model-wideres-epoch76.pt'
net = nn.DataParallel(WideResNet()).cuda()
net.load_state_dict(torch.load(ckpt))
net.eval()
print(ckpt)

cudnn.benchmark = True

# PGD Attack
def _pgd_whitebox(model, X, y, epsilon, num_steps=args.num_steps, step_size=args.step_size):
# def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    _, out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd)[1], y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    # err_pgd = (model(X_pgd)[1].data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return X_pgd

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 # color=plt.cm.Set1(label[i] / 10.),
                 color=plt.cm.Set3(label[i]),
                 fontdict={'weight': 'bold', 'size': 7})
        plt.plot(data[i, 0], data[i, 1])
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
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
            # 对 benign data 生成 representation
            # out4, outputs = net(inputs)
            # out4 = out4.reshape(bs, -1)

            # 生成 adv data，然后得到 representation
            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            X_pgd = _pgd_whitebox(net, X, y, epsilon=epsilon)
            out4, outputs = net(X_pgd)
            out4 = out4.reshape(bs, -1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # T-sne 可视化
            # tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=100, init='pca', random_state=0)
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time()
            out4 = out4.cpu().numpy()
            targets = targets.cpu().numpy()

            result = tsne.fit_transform(out4)
            fig = plot_embedding(result, targets, 't-SNE embedding of the CIFAR-10 (time %.2fs)' % (time() - t0))
            break
    # Save checkpoint.
    acc = 100.*correct/total
    return acc

test()


# print("model best acc：%f ,epoch %d" % (best_acc, best_epoch))
# print(args)