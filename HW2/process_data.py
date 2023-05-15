import os
import torch


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x
    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data():
    feat_dir = './libriphone/feat'
    phone_path = './libriphone'
    concat_nframes = 19
    train_ratio = 0.8

    label_dict = {}
    phone_file = open(os.path.join(phone_path, f'train_labels.txt')).readlines()
    for line in phone_file:
        line = line.strip('\n').split(' ')
        label_dict[line[0]] = [int(p) for p in line[1:]]

    usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
    percent = int(len(usage_list) * train_ratio)
    train_usage_list = usage_list[:percent]
    val_usage_list = usage_list[percent:]
    test_usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    train_usage_list = [line.strip('\n') for line in train_usage_list]
    val_usage_list = [line.strip('\n') for line in val_usage_list]
    test_usage_list = [line.strip('\n') for line in test_usage_list]

    max_len = 3000000
    train_X = torch.empty(max_len, 39 * concat_nframes)
    train_y = torch.empty(max_len, dtype=torch.long)
    val_X = torch.empty(max_len, 39 * concat_nframes)
    val_y = torch.empty(max_len, dtype=torch.long)
    test_X = torch.empty(max_len, 39 * concat_nframes)

    idx = 0
    for i, fname in enumerate(train_usage_list):
        feat = torch.load(os.path.join(feat_dir, 'train', f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        label = torch.LongTensor(label_dict[fname])
        train_X[idx: idx + cur_len, :] = feat
        train_y[idx: idx + cur_len] = label
        idx += cur_len
    train_X = train_X[:idx, :]
    train_y = train_y[:idx]
    idx = 0
    for i, fname in enumerate(val_usage_list):
        feat = torch.load(os.path.join(feat_dir, 'train', f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        label = torch.LongTensor(label_dict[fname])
        val_X[idx: idx + cur_len, :] = feat
        val_y[idx: idx + cur_len] = label
        idx += cur_len
    val_X = val_X[:idx, :]
    val_y = val_y[:idx]
    idx = 0
    for i, fname in enumerate(test_usage_list):
        feat = torch.load(os.path.join(feat_dir, 'test', f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        test_X[idx: idx + cur_len, :] = feat
        idx += cur_len
    test_X = test_X[:idx, :]
    return train_X, train_y, val_X, val_y, test_X
