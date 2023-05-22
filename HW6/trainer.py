import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model import Generator, Discriminator
import glob
from dataset import MyDataset, get_transform


class TrainerGAN:
    def __init__(self, config):
        self.config = config
        self.G = Generator(100)
        self.D = Discriminator(3)
        self.loss = nn.BCELoss()
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.dataloader = None
        self.ckpt_dir = './checkpoints'
        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()

    def prepare_environment(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        names = glob.glob(os.path.join(self.config["data_dir"], '*'))
        transform = get_transform()
        dataset = MyDataset(names, transform)
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()

    def gp(self, r_imgs, f_imgs):
        Tensor = torch.cuda.FloatTensor
        alpha = Tensor(np.random.random((r_imgs.size(0), 1, 1, 1)))
        interpolates = (alpha * r_imgs + (1 - alpha) * f_imgs).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        fake = Variable(Tensor(r_imgs.shape[0]).fill_(1.0), requires_grad=False)
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(1, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self):
        self.prepare_environment()

        for epoch in range(self.config["epochs"]):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {epoch}")
            for i, data in enumerate(progress_bar):
                imgs = data.cuda()
                bs = imgs.size(0)
                z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                r_imgs = Variable(imgs).cuda()
                f_imgs = self.G(z)
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)
                gradient_penalty = self.gp(r_imgs, f_imgs)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                if self.steps % self.config["n_critic"] == 0:
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    f_imgs = self.G(z)
                    f_logit = self.D(f_imgs)
                    loss_G = -torch.mean(f_logit)
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.config["save_dir"], f'Epoch_{epoch:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            self.G.train()
            if epoch  % 5 == 0:
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{epoch}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{epoch}.pth'))
