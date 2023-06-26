import os
import shutil
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
cifar_10_mean = (0.491, 0.482, 0.447)
cifar_10_std = (0.202, 0.199, 0.201)
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)
epsilon1 = 8 / 255 / std
alpha1 = 0.8 / 255 / std


def fgsm(model, x, y, loss_fn, epsilon=epsilon1):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    loss = loss_fn(model(x_adv), y)
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


def ifgsm(model, x, y, loss_fn, epsilon=epsilon1, alpha=alpha1, num_iter=20):
    x_adv = x
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
    return x_adv


def mifgsm(model, x, y, loss_fn, epsilon=epsilon1, alpha=alpha1, num_iter=20, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().to(device)
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum + grad / (grad.abs().sum() + 1e-8)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
    return x_adv


def dmi_mifgsm(model, x, y, loss_fn, epsilon=epsilon1, alpha=alpha1, num_iter=50, decay=1.0, p=0.5):
    x_adv = x
    momentum = torch.zeros_like(x).detach().to(device)
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv_raw = x_adv.clone()
        if torch.rand(1).item() >= p:
            rand = torch.randint(29, 33, (1,)).item()
            x_adv = transforms.Resize((rand, rand))(x_adv)
            left = torch.randint(0, 32 - rand + 1, (1,)).item()
            top = torch.randint(0, 32 - rand + 1, (1,)).item()
            right = 32 - rand - left
            bottom = 32 - rand - top
            x_adv = transforms.Pad([left, top, right, bottom])(x_adv)
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum + grad / (grad.abs().sum() + 1e-8)
        momentum = grad
        x_adv = x_adv_raw + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
    return x_adv


def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)


def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn)
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        adv_ex = ((x_adv) * std + mean).clamp(0, 1)
        adv_ex = (adv_ex * 255).clamp(0, 255)
        adv_ex = adv_ex.detach().cpu().data.numpy().round()
        adv_ex = adv_ex.transpose((0, 2, 3, 1))
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)


def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8))
        im.save(os.path.join(adv_dir, name))
