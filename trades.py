import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
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
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            _, logits = model(adv)
            _, logits_x = model(x_natural)
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(logits, dim=1),
                                           F.softmax(logits_x, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


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

# 针对特定 label 的 adv 做 augment（或者 perturb_steps，step_size 等超参数变化做 aug）
def trades_loss_aug(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    # train (perturb_steps=10，step_size=0.007)，test (20，0.003)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_natural_aug = x_natural[0]
    x_adv_aug = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
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

        # 生成 aug 的 adv data
        for _ in range(perturb_steps):
            x_adv_aug.requires_grad_()
            with torch.enable_grad():
                _, out_adv = model(x_adv_aug)
                _, out_nat = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1), F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv_aug])[0]

            # loss_kl.backward()
            # grad = x_adv_aug.grad

            x_adv_aug = x_adv_aug.detach() + step_size * torch.sign(grad.detach())
            x_adv_aug = torch.min(torch.max(x_adv_aug, x_natural - epsilon), x_natural + epsilon)
            x_adv_aug = torch.clamp(x_adv_aug, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv_aug = Variable(torch.clamp(x_adv_aug, 0.0, 1.0), requires_grad=False)
    # 选择特定 label 的 data 送入后续计算 adv loss

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits_x = model(x_natural)
    _, logits_adv = model(x_adv)
    _, logits_adv_aug = model(x_adv_aug)
    logits_x_aug = logits_x[0]
    loss_natural = F.cross_entropy(logits_x, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_x, dim=1))
    loss_robust_aug = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv_aug, dim=1), F.softmax(logits_x, dim=1))
    loss = loss_natural + beta * loss_robust + beta * loss_robust_aug
    return loss