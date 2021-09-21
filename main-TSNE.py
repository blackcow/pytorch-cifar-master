'''Train CIFAR10 with PyTorch
load 模型 test 指标，进行 T-SNE 可视化
可以 load 全部 data，然后可视化
'''
# python main-TSNE.py -dataset CIFAR10 --ckpt --title

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
from models.preactresnet import create_network

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--imlabel', default=5, type=int, help='Label of the remove part of training data')
parser.add_argument('--dele', default=3, type=int, help='Label of the deleted training data')
parser.add_argument('--percent', default=0.1, type=float, help='Percentage of deleted data')
parser.add_argument('--gpu', default='0', type=str, help='GPUs id')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'STL10', 'Imagnette', 'SVHN', 'ImageNet10'], help='train model on dataset')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--ckpt', type=str, default='../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR10/seed1/model-wideres-epoch100.pt',
                    help='model path')
parser.add_argument('--title', type=str, default='ST_CIFAR10_seed1',
                    help='title of save img')

args = parser.parse_args()
print(args)

# 设定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # 对于 TRADES 注释掉
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train_Imagenet10 = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize([96, 96]),
    # transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
bs = args.test_batch_size

if args.dataset == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
# elif args.dataset == 'CIFAR100':
#     testset = cifar10my3.CIFAR100MY(root='../data', train=False, download=True, transform=transform_test, args=args)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
# elif args.dataset == 'Imagnette':
#     testset = ImagenetteTrain('val')
#     # testset = ImagenetteTest()
#     testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
elif args.dataset == 'SVHN':
    testset = torchvision.datasets.SVHN(root='../data', split="test", download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, **kwargs)
elif args.dataset == 'ImageNet10':
    traindir = '../data/ilsvrc2012/train'
    valdir = '../data/ilsvrc2012/val'
    val = torchvision.datasets.ImageFolder(valdir, transform_train_Imagenet10)
    testloader = torch.utils.data.DataLoader(val, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# path = '../Fair-AT/model-cifar-wideResNet/wideresnet/'
# ST
# ckpt = '/hot-data/niuzh/Mycode/Fair-AT/model-cifar-wideResNet/wideresnet' \
#     '/ST-ori/e0.031_depth34_widen10_drop0.0/'
# ICML-21
# ckpt = '../Robust-Fair/cifar10/models-preactresnet/fair1/trade_120_1.0.pt'
# ST SVHN
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_SVHN/kplabel_seed1/percent_1.0/model-wideres-epoch100.pt'
# ST ImageNet-10
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_ImageNet10/kplabel_seed1/percent_1.0/model-wideres-epoch100.pt'
# ST el
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_el_CIFAR10/seed2/model-wideres-epoch100.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_only_el_CIFAR10/seed4/model-wideres-epoch100.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_el_logits_CIFAR10/seed3/model-wideres-epoch76.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_el_li_CIFAR10/seed1/model-wideres-epoch100.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_el_li2_CIFAR10/seed1/model-wideres-epoch100.pt'
# ST CIFAR10 label smooth
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_label_smooth_CIFAR10/seed4/model-wideres-epoch100.pt'
# ST
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR10/seed1/model-wideres-epoch100.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST/rmlabel_1/percent_0.0/model-wideres-epoch100.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_el_li2_CIFAR10/seed5/model-wideres-epoch100.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_label_smooth35_CIFAR10/seed5/model-wideres-epoch100.pt'
#ST reweight
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_reweight_CIFAR10/seed5/model-wideres-epoch100.pt'
# AT reweigth
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_CIFAR10/seed3/model-wideres-epoch76.pt'
#ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/AT_reweight_CIFAR10/seed5/model-wideres-epoch76.pt'
# AT ImageNet-10
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_ImageNet10/seed1/model-wideres-epoch76.pt'
# ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_ImageNet10/seed2/model-wideres-epoch76.pt'
# title = 'ST_CIFAR10_seed1'
# net = WideResNet().cuda()
# net = nn.DataParallel(WideResNet()).cuda()
# net = nn.DataParallel(create_network()).cuda()


title = args.title
ckpt = args.ckpt
if args.dataset == 'CIFAR10' or 'STL10' or 'Imagnette' or 'SVHN' or 'ImageNet10':
    num_classes = 10
elif args.dataset == 'CIFAR100':
    num_classes = 100

net = nn.DataParallel(create_network(num_classes)).cuda()
net.load_state_dict(torch.load(ckpt))
net.eval()
print(ckpt)

cudnn.benchmark = True

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
    if not os.path.exists("./img"):
        os.mkdir('./img')
    img_title = "./img/"+ title + ".png"
    plt.savefig(img_title, dpi=300)
    plt.show()
    return fig

def test():
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    rep = []
    y = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out4, outputs = net(inputs)
            out4 = out4.reshape(len(inputs), -1)
            rep.append(out4)
            y.append(targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # 绘制部分 test data 的分布，暂定 10 个 batch
            # if len(y) > 10:
            #     break
        acc = 100. * correct / total
        print(acc)
        # T-sne 可视化
        rep = torch.cat(rep)
        y = torch.cat(y)
        rep = rep.cpu().numpy()
        y = y.cpu().numpy()
        # tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=100, init='pca', random_state=0)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        t0 = time()


        result = tsne.fit_transform(rep)
        # fig = plot_embedding(result, y, 't-SNE embedding of the CIFAR-10 (time %.2fs)' % (time() - t0))
        fig = plot_embedding(result, y, title)
    # Save checkpoint.

    return acc

acc = test()
# print(acc)

# print("model best acc：%f ,epoch %d" % (best_acc, best_epoch))
# print(args)