import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from Module.set_transformer import PMA
from torch_geometric.nn.inits import reset, uniform
import random


EPS = 1e-15


class POIEncoder(nn.Module):
    """POI GCN encoder"""
    def __init__(self, in_channels, hidden_channels):
        super(POIEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class POI2Region(nn.Module):
    """POI - region aggregation and GCN at regional level"""
    def __init__(self, hidden_channels, num_heads):
        super(POI2Region, self).__init__()
        self.PMA = PMA(dim=hidden_channels, num_heads=num_heads, num_seeds=1, ln=False)
        self.conv = GCNConv(hidden_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, zone, region_adjacency):
        region_emb = x.new_zeros((zone.max()+1, x.size()[1]))
        for index in range(zone.max() + 1):
            poi_index_in_region = (zone == index).nonzero(as_tuple=True)[0]
            region_emb[index] = self.PMA(x[poi_index_in_region].unsqueeze(0)).squeeze()
        region_emb = self.conv(region_emb, region_adjacency)
        region_emb = self.prelu(region_emb)
        return region_emb


def corruption(x):
    """corruption function to generate negative POIs through random permuting POI initial features"""
    return x[torch.randperm(x.size(0))]


class HierarchicalGraphInfomax(torch.nn.Module):
    r"""The Hierarchical Graph Infomax Module for learning region representations"""
    def __init__(self, hidden_channels, poi_encoder, poi2region, region2city, corruption, alpha):
        super(HierarchicalGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.poi_encoder = poi_encoder
        self.poi2region = poi2region
        self.region2city = region2city
        self.corruption = corruption
        self.alpha = alpha
        self.weight_poi2region = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.weight_region2city = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.region_embedding = torch.tensor(0)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.poi_encoder)
        reset(self.poi2region)
        reset(self.region2city)
        uniform(self.hidden_channels, self.weight_poi2region)
        uniform(self.hidden_channels, self.weight_region2city)

    def forward(self, data):
        """forward function to generate POI, region, and city representations"""
        pos_poi_emb = self.poi_encoder(data.x, data.edge_index, data.edge_weight)
        cor_x = self.corruption(data.x)
        neg_poi_emb = self.poi_encoder(cor_x, data.edge_index, data.edge_weight)
        region_emb = self.poi2region(pos_poi_emb, data.region_id, data.region_adjacency)
        self.region_embedding = region_emb
        neg_region_emb = self.poi2region(neg_poi_emb, data.region_id, data.region_adjacency)
        city_emb = self.region2city(region_emb, data.region_area)
        pos_poi_emb_list = []
        neg_poi_emb_list = []
        """hard negative sampling procedure"""
        for region in range(torch.max(data.region_id)+1):
            id_of_poi_in_a_region = (data.region_id == region).nonzero(as_tuple=True)[0]
            poi_emb_of_a_region = pos_poi_emb[id_of_poi_in_a_region]
            hard_negative_choice = random.random()
            if hard_negative_choice < 0.25:
                hard_example_range = ((data.coarse_region_similarity[region] > 0.6) & (data.coarse_region_similarity[region] < 0.8)).nonzero(as_tuple=True)[0]
                if hard_example_range.size()[0] > 0:
                    another_region_id = random.sample(hard_example_range.tolist(), 1)[0]
                else:
                    another_region_id = random.sample((set(data.region_id.tolist()) - set([region])), 1)[0]
            else:
                another_region_id = random.sample((set(data.region_id.tolist())-set([region])), 1)[0]
            id_of_poi_in_another_region = (data.region_id == another_region_id).nonzero(as_tuple=True)[0]
            poi_emb_of_another_region = pos_poi_emb[id_of_poi_in_another_region]
            pos_poi_emb_list.append(poi_emb_of_a_region)
            neg_poi_emb_list.append(poi_emb_of_another_region)
        return pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb

    def discriminate_poi2region(self, poi_emb_list, region_emb, sigmoid=True):
        values = []
        for region_id, region in enumerate(poi_emb_list):
            if region.size()[0] > 0:
                region_summary = region_emb[region_id]
                value = torch.matmul(region, torch.matmul(self.weight_poi2region, region_summary))
                values.append(value)
        values = torch.cat(values, dim=0)
        return torch.sigmoid(values) if sigmoid else values

    def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
        value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb):
        r"""Computes the mutual information maximization objective among the POI-region-city hierarchy."""
        pos_loss_region = -torch.log(
            self.discriminate_poi2region(pos_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
        neg_loss_region = -torch.log(
            1 - self.discriminate_poi2region(neg_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
        pos_loss_city = -torch.log(
            self.discriminate_region2city(region_emb, city_emb, sigmoid=True) + EPS).mean()
        neg_loss_city = -torch.log(
            1 - self.discriminate_region2city(neg_region_emb, city_emb, sigmoid=True) + EPS).mean()
        loss_poi2region = pos_loss_region + neg_loss_region
        loss_region2city = pos_loss_city + neg_loss_city
        return loss_poi2region * self.alpha + loss_region2city * (1 - self.alpha)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)

    def get_region_emb(self):
        return self.region_embedding.clone().cpu().detach()
