

import argparse
import os
from Eval_module.urban_func import uf_inference
from Eval_module.population_density import pd_estimation
from Eval_module.housing_price import hp_estimation


def parse_args():
    """ parsing the arguments that are used in for testing HGI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='shenzhen', help='city name, can be xiamen or shenzhen')
    parser.add_argument('--task', type=str, default='uf', help='can be uf (urban function), pd (population density), '
                                                               'or hp (house price)')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    args = parse_args()
    """
    please note that the urban function inference is using mocked ground truth data, 
    real ground truth data should be requested at
    http://geoscape.pku.edu.cn/index.html
    """
    if args.task == 'uf':
        uf_inference(args.city, args.device)
    elif args.task == 'pd':
        pd_estimation(args.city)
    elif args.task == 'hp':
        hp_estimation(args.city)
