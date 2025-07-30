import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from colorama import Fore, Style, init
import numpy as np
import random

init(autoreset=True)

from solver import Solver

seeds = [2024, 2025, 2026, 2027, 2028]


def set_seed(seed):
    # seeding
    print(Fore.RED + f'Current seed is set as {seed}')
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print("CUDA Available: ", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define (hyper) parameters for PIR NCDE')
    parser.add_argument('--dataset', type=str, default='TAO')
    parser.add_argument('--data_path', type=str, default='../../MTS_IGAD/Datasets/TSB-AD-M')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--hidden_function', type=int, default=256)
    parser.add_argument('--hidden_function_layers', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=256)
    parser.add_argument('--time_head_num', type=int, default=8)
    parser.add_argument('--spatial_head_num', type=int, default=8)
    parser.add_argument('--lambda_derivative_recon', type=float, default=1.0)
    parser.add_argument('--if_tanh', type=bool, default=True)
    parser.add_argument('--if_ts', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--step_size', type=float, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--missing_rate', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    # Load customized settings
    configs = json.load(open(f'./configs/{args.dataset}.json', 'r'))
    print(configs)
    args.hidden_function = configs['hidden_function']
    args.hidden_function_layers = configs['hidden_function_layers']
    args.output_channels = configs['output_channels']
    args.time_head_num = configs['time_head_num']
    args.spatial_head_num = configs['spatial_head_num']
    args.lambda_derivative_recon = configs['lambda_derivative_recon']
    args.if_tanh = configs['if_tanh']
    # args.if_ts = False
    print(args)

    # Create path for saving the final results
    results_save_path = f'./{args.save_path}/{args.dataset}/'
    os.makedirs(results_save_path, exist_ok=True)

    df_records = pd.DataFrame(
        columns=['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1',
                 'Affiliation-F', 'Seed'])

    for seed in seeds:
        # Set seed
        set_seed(seed)

        solver = Solver(args, seed)

        # Train model
        solver.train_model()

        # Test model
        evaluation_result, final_time_factor, final_system_factor = solver.test_model()
        new_records = {
            'AUC-PR': evaluation_result['AUC-PR'],
            'AUC-ROC': evaluation_result['AUC-ROC'],
            'VUS-PR': evaluation_result['VUS-PR'],
            'VUS-ROC': evaluation_result['VUS-ROC'],
            'Standard-F1': evaluation_result['Standard-F1'],
            'PA-F1': evaluation_result['PA-F1'],
            'Event-based-F1': evaluation_result['Event-based-F1'],
            'R-based-F1': evaluation_result['R-based-F1'],
            'Affiliation-F': evaluation_result['Affiliation-F'],
            'Seed': seed
        }
        with open(f"{results_save_path}/evaluation_results_{args.dataset}_{str(args.missing_rate)}_{str(seed)}.json", "w", encoding="utf-8") as file:
            json.dump(new_records, file, indent=4)
        file.close()
        df_records = pd.concat([df_records, pd.DataFrame([new_records])], ignore_index=True)
        print(evaluation_result)

        with open(f'./results/{args.dataset}_factor_{str(args.missing_rate)}.txt', 'a') as file:
            file.write(f'Seed: {seed}, Time Factor: {final_time_factor}, System Factor: {final_system_factor}\n')
        file.close()

    # Save results
    df_records.to_csv(results_save_path + f'{args.dataset}_results_{str(args.missing_rate)}.csv', index=False)
