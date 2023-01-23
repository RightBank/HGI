import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import random
import os
"""this script is used to evaluate the performance in the task of urban function inference."""


class Net(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=512):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, data):
        return self.net(data)


class Evaluator:
    def __init__(self, embeddings, labels, split, device, seed=None):
        self.embeddings = embeddings
        self.labels = labels
        self.size = embeddings.shape[0]
        self.split = split
        self.device = device
        self.model = Net(embeddings.shape[1], labels.shape[1]).to(self.device)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.999)
        idxs = np.arange(self.size)
        if seed is None:
            seed = np.random.randint(0, 100)
        mix, self.test_index = train_test_split(idxs, test_size=split[2] / (split[0] + split[1] + split[2]), random_state=seed)
        self.train_index, self.val_index = train_test_split(mix, test_size=split[1] / (split[0] + split[1]), random_state=seed)
        self.test_label = self.labels[self.test_index]
        self.train_data = torch.utils.data.TensorDataset(torch.from_numpy(self.embeddings[self.train_index]),
                                                         torch.from_numpy(self.labels[self.train_index]))
        self.val_data = torch.utils.data.TensorDataset(torch.from_numpy(self.embeddings[self.val_index]),
                                                       torch.from_numpy(self.labels[self.val_index]))
        self.test_data = torch.utils.data.TensorDataset(torch.from_numpy(self.embeddings[self.test_index]),
                                                        torch.from_numpy(self.test_label))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=512, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=512, shuffle=False)

    def train(self, epochs):
        lowest_val_loss = float('inf')
        best_test_kl_div = 0
        best_test_l1 = 0
        best_test_cos = 0
        for epoch in trange(epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                train_loss = self.loss(output.log(), target)
                train_loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_size = 0
                val_total_loss = 0
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_total_loss += self.loss(output.log(), target).item() * data.shape[0]
                    val_size += data.shape[0]
                val_loss = val_total_loss / val_size
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    test_size = 0
                    test_kl_div_total = 0
                    test_l1_total = 0
                    test_cos_total = 0
                    for data, target in self.test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        test_kl_div_total += F.kl_div(output.log(), target, reduction='batchmean').item() * data.shape[
                            0]
                        test_l1_total += torch.sum(torch.abs(output - target)).item()
                        test_cos_total += torch.sum(torch.cosine_similarity(output, target, dim=1)).item()
                        test_size += data.shape[0]
                    best_test_kl_div = test_kl_div_total / test_size
                    best_test_l1 = test_l1_total / test_size
                    best_test_cos = test_cos_total / test_size

        print('Best test_loss: {:.4f}, test_l1: {:.4f}, test_cos: {:.4f}'.format(best_test_kl_div, best_test_l1,
                                                                                best_test_cos))
        return best_test_kl_div, best_test_l1, best_test_cos


def uf_inference(city, device):
    warnings.warn("You are using mocked ground truth data. The real ground truth data should be requested at http://geoscape.pku.edu.cn/index.html")
    if city == 'xiamen':
        hgi_file_name = "./Emb/xiamen_emb"
        ground_truth_file_name = "./Data/ground_truth/xiamen/mocked_xiamen_uf_ground_truth"
    elif city == 'shenzhen':
        hgi_file_name = "./Emb/shenzhen_emb"
        ground_truth_file_name = "./Data/ground_truth/shenzhen/mocked_shenzhen_uf_ground_truth"

    region_emb = torch.load(hgi_file_name)
    with open(ground_truth_file_name, 'rb') as handle:
        ground_truth_dict = pickle.load(handle)
    emb_list = []
    y_list = []
    for zone, emb in enumerate(region_emb):
        if zone in ground_truth_dict:
            emb_list.append(region_emb[zone].tolist())
            y_list.append(ground_truth_dict[zone])
    embeddings = np.asarray(emb_list, dtype=np.float32)
    labels = np.asarray(y_list, dtype=np.float32)
    kl_div, l1, cos = [], [], []
    for i in range(10):
        evaluator = Evaluator(embeddings, labels, [0.6, 0.2, 0.2], device)
        test_kl_div, test_l1, test_cos = evaluator.train(100)
        kl_div.append(test_kl_div)
        l1.append(test_l1)
        cos.append(test_cos)
    print('Result for ')
    print('=============Result Table=============')
    print('L1\tstd\tKL-Div\tstd\tCosine\tstd')
    print('{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}'.format(np.mean(l1), np.std(l1), np.mean(kl_div), np.std(kl_div),
                                                                    np.mean(cos), np.std(cos)))

