import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from math import isnan

"""this script is used to evaluate the performance in the task of housing price prediction in xiamen."""


def hp_estimation(city):
    print("Evaluating housing price estimation in {} for ten times".format(city))

    if city == 'xiamen':
        dataset = pd.read_csv('Data/ground_truth/xiamen/housing_price_xiamen.csv')
        zone_emb = torch.load('Emb/xiamen_emb')

    elif city == 'shenzhen':
        dataset = pd.read_csv('Data/ground_truth/shenzhen/housing_price_shenzhen.csv')
        zone_emb = torch.load('Emb/shenzhen_emb')

    zone_list = list(zip(zone_emb.tolist(), list(dataset['housing_price'])))
    removed_zone_list = []
    for zone in zone_list:
        if not isnan(zone[1]):
            removed_zone_list.append(zone)
    zone_list = removed_zone_list

    rmse_list = []
    r2_list = []
    mae_list = []

    for iter in range(10):
        print("iter: ", iter)
        np.random.shuffle(zone_list)
        x_list = [zone[0] for zone in zone_list]
        y_list = [zone[1] for zone in zone_list]
        x_train = np.array(x_list[:int(len(x_list) * 0.8)])
        y_train = np.array(y_list[:int(len(y_list) * 0.8)])
        rf = RandomForestRegressor(random_state=iter)
        rf.fit(x_train, y_train)
        x_test = np.array(x_list[int(len(x_list) * 0.8):])
        y_test = np.array(y_list[int(len(y_list) * 0.8):])
        y_pred = rf.predict(x_test)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        rmse_list.append(rmse)
        print("rmse:", rmse)
        r2 = metrics.r2_score(y_test, y_pred)
        r2_list.append(r2)
        print("r2:", r2)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
        print("mae:", mae)
    average_MAE = np.mean(mae_list)
    std_MAE = np.std(mae_list)
    average_RMSE = np.mean(rmse_list)
    std_RMSE = np.std(rmse_list)
    average_R2 = np.mean(r2_list)
    std_R2 = np.std(r2_list)
    print("Result of housing price estimation in {}:".format(city))
    print('=============Result Table=============')
    print('MAE\t\tstd\t\tRMSE\t\tstd\t\tR2\t\tstd')
    print(f'{average_MAE}\t{std_MAE}\t{average_RMSE}\t{std_RMSE}\t{average_R2}\t{std_R2}')