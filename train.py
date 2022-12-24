"""
@file: train.py
@time: 2022/11/5
@project: HGI
@description: main function to run HGI
@author: Weiming Huang, Daokun Zhang, Gengchen Mai
@contact: weiming.huang@ntu.edu.sg
"""
import argparse
from Module.city_data import hgi_graph
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from Module.hgi_module import *
from tqdm import trange
import pytorch_warmup as warmup
import math
import os


def parse_args():
    """ parsing the arguments that are used in HGI """
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='shenzhen', help='city name, such as shenzhen')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of output representation')
    parser.add_argument('--alpha', type=float, default=0.5, help='the hyperparameter to balance mutual information')
    parser.add_argument('--attention_head', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.006)
    parser.add_argument('--max_norm', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--warmup_period', type=int, default=40)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_name', type=str, default='shenzhen_emb')
    return parser.parse_args()


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    args = parse_args()
    """load the graph data of a study area"""
    data = hgi_graph(args.city).to(args.device)
    """load the Module"""
    model = HierarchicalGraphInfomax(
        hidden_channels=args.dim,
        poi_encoder=POIEncoder(data.num_features, args.dim),
        poi2region=POI2Region(args.dim, args.attention_head),
        region2city=lambda z, area: torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1)),
        corruption=corruption,
        alpha=args.alpha,
    ).to(args.device)
    """load the optimizer, scheduler (including a warmup scheduler)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, verbose=False)
    warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_period)


    def train():
        model.train()
        optimizer.zero_grad()
        pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb = model(data)
        loss = model.loss(pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        with warmup_scheduler.dampening():
            scheduler.step()
        return loss.item()

    print("Start training region embeddings for the city of {}".format(args.city))
    t = trange(1, args.epoch + 1)
    lowest_loss = math.inf
    region_emb_to_save = torch.FloatTensor(0)
    for epoch in t:
        loss = train()
        if loss < lowest_loss:
            """save the embeddings with the lowest loss"""
            region_emb_to_save = model.get_region_emb()
            lowest_loss = loss
        t.set_postfix(loss='{:.4f}'.format(loss), refresh=True)
    torch.save(region_emb_to_save, f'./Emb/{args.save_name}')
    print(f"Region embeddings of {args.city} has been save to ./Emb/{args.save_name}")










