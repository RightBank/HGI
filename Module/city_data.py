import pickle as pkl
import torch
from torch_geometric.data import Data
import os


def hgi_graph(city_name='shenzhen'):
    """the function hgi_graph() returns a Data object of the city graph.
    Note that the data file should be put under the directory ./Data/"""
    city_dict_file = f'./Data/{city_name}_data.pkl'
    if city_name == "shenzhen" and not os.path.exists(city_dict_file):
        print("Please download data for Shenzhen from https://figshare.com/articles/dataset/Sub-sampled_dataset_for_Shenzhen_HGI_region_embedding_example_dataset_/21836496")
        exit()
    elif not os.path.exists(city_dict_file):
        print(f"Please construct dataset for {city_name} following the instructions in README.md, and making it in "
              f"the same structure of the data for Shenzhen, which can be downloaded from "
              f"https://figshare.com/articles/dataset/Sub-sampled_dataset_for_Shenzhen_HGI_region_embedding_example_dataset_/21836496")
        exit()
    with open(city_dict_file, 'rb') as handle:
        city_dict = pkl.load(handle)
    poi_graph = Data(x=torch.tensor(city_dict['node_features'], dtype=torch.float32),
                     edge_index=torch.tensor(city_dict['edge_index'], dtype=torch.int64),
                     edge_weight=torch.tensor(city_dict['edge_weight'], dtype=torch.float32),
                     region_id=torch.tensor(city_dict['region_id'], dtype=torch.int64),
                     region_area=torch.tensor(city_dict['region_area'], dtype=torch.float32),
                     coarse_region_similarity=torch.tensor(city_dict['coarse_region_similarity'],
                                                           dtype=torch.float32),
                     region_adjacency=torch.tensor(city_dict['region_adjacency'], dtype=torch.int64))
    return poi_graph

