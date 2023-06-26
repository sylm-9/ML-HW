import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from attack_func import epoch_benign, gen_adv_examples, ifgsm, mifgsm, dmi_mifgsm, create_dir
from model import ensembleNet
from dataset import AdvDataset
from resnet import ResNet18
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorchcv.model_provider import get_model as ptcv_get_model
import numpy as np
from PIL import Image

def saveimg(attack_name, model):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 20))
    cnt = 0
    for i, cls_name in enumerate(classes):
        path = f'{cls_name}/{cls_name}1.png'
        cnt += 1
        plt.subplot(len(classes), 4, cnt)
        im = Image.open(f'./data/{path}')
        logit = model(transform(im).unsqueeze(0).to(device))[0]
        predict = logit.argmax(-1).item()
        prob = logit.softmax(-1)[predict].item()
        plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
        plt.axis('off')
        plt.imshow(np.array(im))

        cnt += 1
        plt.subplot(len(classes), 4, cnt)
        im = Image.open(f'./ensemble_{attack_name}/{path}')
        logit = model(transform(im).unsqueeze(0).to(device))[0]
        predict = logit.argmax(-1).item()
        prob = logit.softmax(-1)[predict].item()
        plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
        plt.axis('off')
        plt.imshow(np.array(im))
    plt.tight_layout()
    plt.savefig(attack_name+'.png')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    cifar_10_mean = (0.491, 0.482, 0.447)
    cifar_10_std = (0.202, 0.199, 0.201)
    mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
    std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)
    epsilon = 8 / 255 / std
    alpha = 0.8 / 255 / std
    root = './data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_10_mean, cifar_10_std)
    ])

    adv_set = AdvDataset(root, transform=transform)
    adv_names = adv_set.__getname__()
    adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)

    model_names = [
        'nin_cifar10',
        'resnet20_cifar10',
        'preresnet20_cifar10'
    ]
    resnet18 = ResNet18().cuda()
    state_dict = torch.load('ckpt.pth')['net']
    resnet18.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    ensemble_model = ensembleNet(model_names, resnet=resnet18).to(device)
    ensemble_model.eval()

    benign_acc, benign_loss = epoch_benign(ensemble_model, adv_loader, loss_fn)
    print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

    adv_examples, acc, loss = gen_adv_examples(ensemble_model, adv_loader, ifgsm, loss_fn)
    print(f'ensemble_ifgsm_acc = {acc:.5f}, ensemble_ifgsm_loss = {loss:.5f}')
    create_dir(root, 'ensemble_ifgsm', adv_examples, adv_names)
    saveimg("ifgsm",model)

    adv_examples, acc, loss = gen_adv_examples(ensemble_model, adv_loader, mifgsm, loss_fn)
    print(f'ensemble_mifgsm_acc = {acc:.5f}, ensemble_mifgsm_loss = {loss:.5f}')
    create_dir(root, 'ensemble_mifgsm', adv_examples, adv_names)
    saveimg("mifgsm", model)

    adv_examples, acc, loss = gen_adv_examples(ensemble_model, adv_loader, dmi_mifgsm, loss_fn)
    print(f'ensemble_dmi_mifgsm_acc = {acc:.5f}, ensemble_dim_mifgsm_loss = {loss:.5f}')
    create_dir(root, 'ensemble_dmi_mifgsm', adv_examples, adv_names)
    saveimg("dmi_mifgsm", model)



