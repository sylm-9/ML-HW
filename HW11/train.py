import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import cv2
from model import DomainClassifier, LabelPredictor, FeatureExtractor
import matplotlib.pyplot as plt
from sklearn import manifold

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 200
    source_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        transforms.ToTensor()
    ])
    source_dataset = ImageFolder(
        './real_or_drawing/train_data', transform=source_transform)
    target_dataset = ImageFolder(
        './real_or_drawing/test_data', transform=target_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())

    for epoch in range(epochs):
        lamb = np.log(1.02+1.7*epoch/epochs)
        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0
        feature_extractor.train()
        label_predictor.train()
        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

            source_data = source_data.to(device)
            source_label = source_label.to(device)
            target_data = target_data.to(device)
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros(
                [source_data.shape[0] + target_data.shape[0], 1]).to(device)
            domain_label[:source_data.shape[0]] = 1
            feature = feature_extractor(mixed_data)
            domain_logits = domain_classifier(feature.detach())
            loss = domain_criterion(domain_logits, domain_label)
            running_D_loss += loss.item()
            loss.backward()
            optimizer_D.step()

            class_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            loss = class_criterion(class_logits, source_label) - \
                lamb * domain_criterion(domain_logits, domain_label)
            running_F_loss += loss.item()
            loss.backward()
            optimizer_F.step()
            optimizer_C.step()
            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()
            total_hit += torch.sum(torch.argmax(class_logits,
                                   dim=1) == source_label).item()
            total_num += source_data.shape[0]
            print(i, end='\r')
        train_D_loss, train_F_loss, train_acc = running_D_loss / \
            (i+1), running_F_loss/(i+1), total_hit/total_num
        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(
            epoch, train_D_loss, train_F_loss, train_acc))
    torch.save(feature_extractor.state_dict(), 'extractor_model.bin')
    torch.save(label_predictor.state_dict(), 'predictor_model.bin')
