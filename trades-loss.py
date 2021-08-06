import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pgd import attack_pgd

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

# 返回 nature 和 boundary loss
def boundary_loss_test(model, x_natural, y, optimizer=None, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            # loss_kl.backward()
            # grad = x_adv.grad

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y) * len(y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    # loss = loss_natural + beta * loss_robust
    return loss_natural, loss_robust

# 使用 PGD loss 进行计算，返回 nature 和 robust loss
def robust_loss_test(model, x_natural, y, optimizer=None, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            # loss_kl.backward()
            # grad = x_adv.grad

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # TRADES 的 attack
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # PGD 的 attack
    x_adv_pgd = _pgd_whitebox(model, x_natural, y, epsilon=epsilon, AT_method='AT', num_steps=perturb_steps, step_size=step_size)
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    _, logits_adv_pgd = model(x_adv_pgd)
    loss_natural = F.cross_entropy(logits_x, y) * len(y)
    loss_robust = F.cross_entropy(logits_adv, y) * len(y)
    loss_robust_pgd = F.cross_entropy(logits_adv_pgd, y) * len(y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    # loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    # loss = loss_natural + beta * loss_robust
    # return loss_natural, loss_robust
    return loss_robust_pgd, loss_robust

def _pgd_whitebox(model, X, y, epsilon, AT_method, num_steps, step_size):
    rep, out = model(X)
    random = True
    N, C, H, W = rep.size()
    rep = rep.reshape([N, -1])
    out = out.data.max(1)[1]
    if AT_method == 'ST':
        return out, rep, out, rep
    elif AT_method == 'AT':
        X_pgd = Variable(X.data, requires_grad=True)
        if random:
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
        return X_pgd