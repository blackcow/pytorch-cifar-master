'''Train CIFAR10 with PyTorch
针对 3，5 data，找到误分类的 idx，重复情况如何
测试 benign acc 和 robust acc（在各个 label 下）,不受限于各 label 的 data 数量
仅测试，没有 rep 表示.
需要选择测试 model 为 AT 还是 ST，来确定 PGD attack 是否使用
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import random
import os
import argparse

from models import *
# from utils import progress_bar
# from network import create_network

import cifar10my2
import cifar10my3
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from time import time
import numpy as np
import matplotlib.pyplot as plt
from models.wideresnet import WideResNet
from models.densenet import DenseNet121
from models.preactresnet import create_network
from torch.autograd import Variable
from time import time
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary
from dataset.imagnette import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gpu', default='0', type=str, help='GPUs id')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# Model facotrs
parser.add_argument('--depth', type=int, default=34, metavar='N',
                    help='model depth (default: 34)')
parser.add_argument('--widen_factor', type=int, default=10, metavar='N',
                    help='model widen_factor (default: 10)')
parser.add_argument('--droprate', type=float, default=0.0, metavar='N',
                    help='model droprate (default: 0.0)')
# draw imgs
parser.add_argument('--factors', default='model', type=str, metavar='N',
                    choices=['widen_factor', 'depth', 'droprate', 'epsilon', 'model'],
                    help='tensorboard draw img factors')

# PGD attack
parser.add_argument('--epsilon', default=0.031, type=float, help='perturbation')
parser.add_argument('--num-steps', default=20, help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, help='perturb step size')
parser.add_argument('--random', default=True, help='random initialization for PGD')
# test on dataset
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'STL10', 'Imagnette', 'SVHN', 'ImageNet10'],
                    help='train model on dataset')
# 选择测试 ST model 还是 AT
parser.add_argument('--AT-method', type=str, default='ST',
                    help='AT method', choices=['AT', 'ST'])
parser.add_argument('--ckpt', type=str, default='', help='model dir')
args = parser.parse_args()
print(args)

# 设定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # 对于 TRADES 提供的 model 注释掉
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train_Imagenet10 = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize([96, 96]),
    transforms.ToTensor(),
])
use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
bs = args.test_batch_size
if args.dataset == 'CIFAR10':
    testset = cifar10my3.CIFAR10MY(root='../data', train=False, download=True, transform=transform_test, args=args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
elif args.dataset == 'CIFAR100':
    testset = cifar10my3.CIFAR100MY(root='../data', train=False, download=True, transform=transform_test, args=args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
elif args.dataset == 'Imagnette':
    testset = ImagenetteTrain('val')
    # testset = ImagenetteTest()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
elif args.dataset == 'SVHN':
    testset = torchvision.datasets.SVHN(root='../data', split="test", download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, **kwargs)
elif args.dataset == 'ImageNet10':
    traindir = '../data/ilsvrc2012/train'
    valdir = '../data/ilsvrc2012/val'
    val = torchvision.datasets.ImageFolder(valdir, transform_train_Imagenet10)
    testloader = torch.utils.data.DataLoader(val, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
cudnn.benchmark = True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def loadmodel_preactresnte(i, ckpt, factor):
    print('==> Building model..')
    # AT preactresnet
    ckptlist =['../Fair-AT/model-cifar-wideResNet/preactresnet/ST_el_li2_CIFAR10/seed4/model-wideres-epoch100.pt',
    #        '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_label_smooth_CIFAR10/seed1/model-wideres-epoch100.pt',
           '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR10/seed1/model-wideres-epoch100.pt'
           ]
    # ckptlist = ['../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR10/seed1/model-wideres-epoch100.pt',
    # '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR10/seed2/model-wideres-epoch100.pt']
    # '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR10/seed3/model-wideres-epoch100.pt']
    ckpt = ckptlist[i]
    # ckpt_list = ['model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']

    if args.dataset == 'CIFAR10' or 'STL10' or 'Imagnette' or 'SVHN' or 'ImageNet10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    net = nn.DataParallel(create_network(num_classes)).cuda()
    net.load_state_dict(torch.load(ckpt))

    # for AT-opt & Fine-tune model
    # checkpoint = torch.load(ckpt)
    # net.load_state_dict(checkpoint['net'])
    net.eval()
    print(ckpt)
    return net

# PGD Attack
def _pgd_whitebox(model, X, y, epsilon, AT_method, num_steps=args.num_steps, step_size=args.step_size, ):
    ori_rep, out_logit = model(X)
    # N, C, H, W = rep.size()
    # rep = rep.reshape([N, -1])
    # out = out_ori.data.max(1)[1]
    if AT_method == 'ST':
        # logits = out_ori.softmax(dim=-1)
        return out_logit, ori_rep
    elif AT_method == 'AT':
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
        pgd_rep, out_pgd = model(X_pgd)
        # out_pgd = out_pgd.data.max(1)[1]
        return out_logit, out_pgd


# input: tensorboard, model, model_name
def test(writer, net, model_name, epsilon, AT_method):
    global best_acc
    global best_epoch

    target = []
    output_rep = []
    output_logit = []

    with torch.no_grad():
        # for inputs, targets in testloader:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            out_logit, ori_rep = _pgd_whitebox(net, X, y, epsilon=epsilon, AT_method=AT_method)
            output_rep.append(ori_rep)
            output_logit.append(out_logit)
            target.append(y)

        # 所有 y，最后一行可能不满一列的长度，单独 concat
        target_tmp = torch.stack(target[:-1])
        target = torch.cat((target_tmp.reshape(-1), target[-1]), dim=0)

        # 统计得到 rep 和输出的 label
        tmp = torch.cat(output_rep[:-1])
        output_rep = torch.cat((tmp, output_rep[-1]), dim=0)
        output_rep = F.adaptive_avg_pool2d(output_rep, (1, 1)).squeeze()
        output_rep = F.normalize(output_rep, dim=1)  # rep 合并后归一化
        tmp = torch.cat(output_logit[:-1])
        output_logit = torch.cat((tmp, output_logit[-1]), dim=0)
        output = output_logit.data.max(1)[1]

        # 分别统计 3,5 label 的 logits
        out_rep_35 = []
        idx_35 = []
        out_35 = []
        target_35 = []
        for i in [3, 5]:
            idx = (target == i).nonzero().flatten()
            idx_35.append(idx)
            # len_idx = len(idx)
            out_rep_35.append(torch.index_select(output_rep, 0, idx))  # 3，5 data output 对应的 rep
            out_35.append(torch.index_select(output, 0, idx))  # 3,5 data output 的 label
            target_35.append(torch.index_select(target, 0, idx))  # 3,5 data output 的 GT label
            # out_35
        #     benign_logits_perlabel = (torch.sum(benign_logits_perlabel, dim=0)/len_idx).cpu().numpy()
        #     benign_logits_avg.append([float('{:.3f}'.format(i)) for i in benign_logits_perlabel])
        # for x in benign_logits_avg:
        #     print(*x)

        # 统计每个 label 正、误分类的结果
        # 统计 3，5 误分类的 idx
        right_idx, wrong_idx = [], []
        right_rep, wrong_rep = [], []

        wrong_idx3 = (out_35[0] != target_35[0]).nonzero().flatten().cpu().numpy()
        wrong_idx5 = (out_35[1] != target_35[1]).nonzero().flatten().cpu().numpy()

        print()
    return wrong_idx3, wrong_idx5

def count(wrong_idx_all):
    y = np.concatenate([wrong_idx_all[0], wrong_idx_all[1], wrong_idx_all[2]], axis=0)
    y = sorted(y)
    # print(y)
    # 统计出现的元素有哪些
    unique_data = np.unique(y)
    # print(unique_data)

    # 统计某个元素出现的次数
    resdata = []
    count = []
    for ii in unique_data:
        resdata.append(y.count(ii))

    for m in np.unique(resdata):
        count.append(resdata.count(m))
    print('len:', len(y))
    print(np.unique(resdata))
    print(count)

def main():
    start = time()
    seed_everything(1)
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # load model
    # 根据测试的 factor 选择对应的 model
    print('factors:', args.factors)
    # logits = [0, 0, 0]
    # logits_robust = [0, 0, 0]
    model_num = 3
    ckpt = args.ckpt
    wrong_idx3_all = []
    wrong_idx5_all = []
    if args.factors == 'model':
        for i in range(model_num):
            print("Test: " + str(i))
            factor = [args.epsilon, args.depth, args.widen_factor, args.droprate]
            # net = loadmodel(i, factor)
            net = loadmodel_preactresnte(i, ckpt, factor)
            # test robust fair model
            # net = loadmodel_robustfair(i, factor)
            wrong_idx3, wrong_idx5 = test(writer, net, 'model_name', factor[0], args.AT_method)
            wrong_idx3_all.append(wrong_idx3)
            wrong_idx5_all.append(wrong_idx5)
    else:
        raise Exception('this should never happen')

    # 统计出现次数
    # from collections import Counter
    # Counter([1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    print('For 3 label:')
    count(wrong_idx3_all)
    print('For 5 label:')
    count(wrong_idx5_all)
    print()

    # sum of the dis of the center rep
    # for m in range(model_num):
    #     print('%.2f' % logits[m])
    # for m in range(model_num):
    #     print('%.2f' % logits_robust[m])

    writer.close()
    end = time()
    print('时间:{:3f}'.format((end - start) / 60))


if __name__ == '__main__':
    main()
