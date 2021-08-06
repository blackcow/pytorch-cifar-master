import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


upper_limit, lower_limit = 1,0
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def normalize(X):
    return (X - mu)/std

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

# 约束 delta + x 后在 [0,1] 区间下
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, step_size, perturb_steps, distance, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    # max_loss = torch.zeros(y.shape[0]).cuda()
    # max_delta = torch.zeros_like(X).cuda()
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X) # 做了啥？
    delta.requires_grad = True
    for _ in range(perturb_steps):
        # output = model(normalize(X + delta))
        output = model(X + delta)
        if early_stop: # 这是啥？
            index = torch.where(output.max(1)[1] == y)[0]
        else:
            index = slice(None, None, None)
        if not isinstance(index, slice) and len(index) == 0:
            break
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        # d = delta[index, :, :, :]
        # g = grad[index, :, :, :]
        # x = X[index, :, :, :]
        delta = torch.clamp(delta + step_size * torch.sign(grad), min=-epsilon, max=epsilon)
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        # delta.data[index, :, :, :] = d
        # 每轮梯度都置零，重新计算新 delta
        delta.grad.zero_()

    # all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
    # max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
    # max_loss = torch.max(max_loss, all_loss)
    max_delta = delta.detach()
    return max_delta

# model, X, y, epsilon, step_size, perturb_steps, restarts, norm,
def pgd_loss(model, X, y, optimizer, epsilon=0.031, step_size=0.003, perturb_steps=10, beta=1.0, distance='l_inf'):
    model.eval()
    delta = attack_pgd(model, X, y, epsilon, step_size, perturb_steps, distance)
    delta = delta.detach()
    # robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
    # x_adv = Variable(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit), requires_grad=False)
    x_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
    model.train()
    # if args.l1:
    #     for name, param in model.named_parameters():
    #         if 'bn' not in name and 'bias' not in name:
    #             robust_loss += args.l1 * param.abs().sum()
    optimizer.zero_grad()
    robust_output = model(x_adv)
    robust_loss = F.cross_entropy(robust_output, y)
    return robust_loss