import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torchcde
import os.path
from colorama import Fore, Style, init
from tqdm import tqdm
import pickle
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc, roc_curve

from evaluation.metrics import get_metrics

init(autoreset=True)

from models.PIR_NCDE import PIR_NCDE
from utils.early_stopping import EarlyStoppingTorch
from utils.slidingWindows import find_length_rank


class Solver(nn.Module):
    def __init__(self, args, seed):
        super(Solver, self).__init__()

        # Define basic parameters
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.batch_size = args.batch_size
        self.window_size = args.window_size
        self.hidden_function = args.hidden_function
        self.hidden_function_layers = args.hidden_function_layers
        self.output_channels = args.output_channels
        self.time_head_num = args.time_head_num
        self.spatial_head_num = args.spatial_head_num
        self.lambda_derivative_recon = args.lambda_derivative_recon
        self.if_tanh = args.if_tanh
        self.if_ts = args.if_ts
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.missing_rate = args.missing_rate
        self.device = torch.device(args.device)
        self.early_stopping = EarlyStoppingTorch(save_path=f'{self.save_path}/{self.dataset}', dataset=self.dataset,
                                                 missing_rate=self.missing_rate, seed=seed, patience=5)
        self.seed = seed
        print(Fore.RED + 'Basic parameters load finished.')

        self.validation_size = 0.2

        # Define required loss functions
        self.loss_function = nn.MSELoss(reduction='none')

        # Load and prepare data
        self.data_preparation()
        print(Fore.RED + f'Data load and prepare finished.')

        # Define model
        self.model = PIR_NCDE(input_channels=self.input_channels, window_size=self.window_size,
                              hidden_function=self.hidden_function, output_channels=self.output_channels,
                              time_head_num=self.time_head_num, spatial_head_num=self.spatial_head_num,
                              hidden_function_layers=self.hidden_function_layers, if_tanh=self.if_tanh,
                              if_ts=self.if_ts).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.schedular = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.step_size,
                                                         gamma=self.gamma)
        print(Fore.RED + 'Model define finished.')

    def train_model(self):
        iteration = 0
        for epoch in range(500):
            # Train stage
            self.model.train()
            loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=True)
            avg_train_loss = 0.0
            for index, (data, coeffs) in loop:
                data = data.to(self.device)
                coeffs = coeffs.to(self.device)

                iteration += 1

                x, recon_time_series, system_derivative, recon_system_derivative, mask, zT = self.model(data, coeffs,
                                                                                                        iteration)

                loss_1 = self.loss_function(x, recon_time_series)
                loss_1 = loss_1.mean(dim=-1) * mask
                loss_2 = self.loss_function(system_derivative, recon_system_derivative) * self.lambda_derivative_recon
                loss_2 = loss_2.mean(dim=-1) * mask
                loss = torch.mean(loss_1 + loss_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item()
                loop.set_description(f"Training Epoch: [{epoch}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_train_loss / (index + 1))

            # Validation stage
            validation_loss = self.validation_model(iteration)
            self.early_stopping(validation_loss, self.model)
            if self.early_stopping.early_stop:
                print(Fore.RED + 'Training finished. Early stop.')
                break

            self.schedular.step()

    def test_model(self):
        # Load optimal parameters
        self.model.load_state_dict(torch.load(
            f'{self.save_path}/{self.dataset}/{self.dataset}_{str(self.missing_rate)}_{str(self.seed)}_best_network.pt'))
        self.model.eval()
        loop = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), leave=True)
        labels_list = []
        time_recon_errors = []
        system_recon_errors = []
        with torch.no_grad():
            for index, (data, coeffs, labels) in loop:
                data = data.to(self.device)
                coeffs = coeffs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.reshape(-1)
                labels_list.append(labels.cpu())

                x, recon_time_series, system_derivative, recon_system_derivative, mask, zT = self.model(data, coeffs, 0)

                x = x.reshape(-1, self.window_size * self.input_channels)
                recon_time_series = recon_time_series.reshape(-1, self.window_size * self.input_channels)
                system_derivative = system_derivative.reshape(-1, self.window_size * self.input_channels)
                recon_system_derivative = recon_system_derivative.reshape(-1, self.window_size * self.input_channels)

                loss_1 = torch.mean(self.loss_function(x, recon_time_series), dim=-1)
                loss_2 = torch.mean(self.loss_function(system_derivative, recon_system_derivative), dim=-1)
                time_recon_errors.append(loss_1.detach().cpu().numpy().reshape(-1))
                system_recon_errors.append(loss_2.detach().cpu().numpy().reshape(-1))

                loop.set_description(f"Validation")

        time_recon_errors = np.concatenate(time_recon_errors, axis=0)
        system_recon_errors = np.concatenate(system_recon_errors, axis=0)
        labels = torch.cat(labels_list, dim=0)
        labels = labels.reshape(-1).numpy()
        final_VUS_PR = -np.inf
        final_results = None
        final_time_factor = None
        final_system_factor = None
        for time_factor in np.linspace(0.0, 1.0, num=11):
            for system_factor in np.linspace(0.0, 1.0, num=11):
                cur_scores = time_factor * time_recon_errors + system_factor * system_recon_errors
                cur_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(cur_scores.reshape(-1, 1)).ravel()
                evaluation_result = get_metrics(cur_scores, labels, slidingWindow=self.slidingWindow,
                                                pred=cur_scores > (np.mean(cur_scores) + 3 * np.std(cur_scores)))
                if evaluation_result['VUS-PR'] > final_VUS_PR:
                    final_VUS_PR = evaluation_result['VUS-PR']
                    final_results = evaluation_result
                    final_time_factor = time_factor
                    final_system_factor = system_factor

        return final_results, final_time_factor, final_system_factor

    def validation_model(self, iteration):
        self.model.eval()
        loop = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), leave=True)
        avg_validation_loss = 0.0
        with torch.no_grad():
            for index, (data, coeffs) in loop:
                data = data.to(self.device)
                coeffs = coeffs.to(self.device)

                x, recon_time_series, system_derivative, recon_system_derivative, mask, zT = self.model(data, coeffs,
                                                                                                        iteration)

                loss_1 = self.loss_function(x, recon_time_series)
                loss_1 = loss_1.mean(dim=-1) * mask
                loss_2 = self.loss_function(system_derivative, recon_system_derivative) * self.lambda_derivative_recon
                loss_2 = loss_2.mean(dim=-1) * mask
                loss = torch.mean(loss_1 + loss_2)

                avg_validation_loss += loss.item()
                loop.set_description(f"Validation")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_validation_loss / (index + 1))

        avg_validation_loss = avg_validation_loss / len(self.val_dataloader)
        return avg_validation_loss

    def data_preparation(self):
        # Load data if they have been processed
        if os.path.exists(f'{self.save_path}/{self.dataset}/train_data_{str(self.missing_rate)}.pt'):
            # Load dataset
            with open(f'{self.data_path}/{self.dataset}.pickle', 'rb') as file:
                dataset_files = pickle.load(file)
            file.close()
            dataset_files = sorted(dataset_files)
            data_file = dataset_files[0]
            df = pd.read_csv(self.data_path + '/' + data_file).dropna()

            # Preprocess data
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            self.slidingWindow = find_length_rank(data, rank=1)

            print(Fore.BLUE + 'Datasets have been processed, load them directly....')
            train_data = torch.load(f'{self.save_path}/{self.dataset}/train_data_{str(self.missing_rate)}.pt')
            train_coeffs = torch.load(f'{self.save_path}/{self.dataset}/train_coeffs_{str(self.missing_rate)}.pt')

            validation_data = torch.load(f'{self.save_path}/{self.dataset}/validation_data_{str(self.missing_rate)}.pt')
            validation_coeffs = torch.load(
                f'{self.save_path}/{self.dataset}/validation_coeffs_{str(self.missing_rate)}.pt')

            test_data = torch.load(f'{self.save_path}/{self.dataset}/test_data_{str(self.missing_rate)}.pt')
            test_coeffs = torch.load(f'{self.save_path}/{self.dataset}/test_coeffs_{str(self.missing_rate)}.pt')
            test_labels = torch.load(f'{self.save_path}/{self.dataset}/test_labels_{str(self.missing_rate)}.pt')

            self.input_channels = train_data.shape[2]

        else:
            # Load dataset
            with open(f'{self.data_path}/{self.dataset}.pickle', 'rb') as file:
                dataset_files = pickle.load(file)
            file.close()
            dataset_files = sorted(dataset_files)
            data_file = dataset_files[0]
            df = pd.read_csv(self.data_path + '/' + data_file).dropna()

            # Preprocess data
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            self.slidingWindow = find_length_rank(data, rank=1)
            train_index = data_file.split('.')[0].split('_')[-3]
            data_train = data[:int(train_index), :]

            # Split data into training, validation, and test sets
            train_data = data_train[:int((1 - self.validation_size) * len(data_train))]
            validation_data = data_train[int((1 - self.validation_size) * len(data_train)):]
            if self.dataset == 'MSL' or self.dataset == 'TAO':
                validation_data = np.vstack([validation_data, validation_data])
            test_data = data

            # Normalize data
            train_data = self.normalize(train_data)
            validation_data = self.normalize(validation_data)
            test_data = self.normalize(test_data)

            # Convert data to PyTorch tensors
            train_data = torch.from_numpy(train_data).float()
            validation_data = torch.from_numpy(validation_data).float()
            test_data = torch.from_numpy(test_data).float()
            test_labels = torch.from_numpy(label).long()

            self.input_channels = train_data.shape[1]

            # Process the original data into batch-based and calculate doefficients in NCDE
            train_data, train_coeffs = self.data_processor(data=train_data, data_type='train')
            validation_data, validation_coeffs = self.data_processor(data=validation_data, data_type='validation')
            test_data, test_coeffs = self.data_processor(data=test_data, data_type='test')

            # Save them for reuse
            torch.save(train_data, f'{self.save_path}/{self.dataset}/train_data_{str(self.missing_rate)}.pt')
            torch.save(train_coeffs, f'{self.save_path}/{self.dataset}/train_coeffs_{str(self.missing_rate)}.pt')

            torch.save(validation_data, f'{self.save_path}/{self.dataset}/validation_data_{str(self.missing_rate)}.pt')
            torch.save(validation_coeffs,
                       f'{self.save_path}/{self.dataset}/validation_coeffs_{str(self.missing_rate)}.pt')

            torch.save(test_data, f'{self.save_path}/{self.dataset}/test_data_{str(self.missing_rate)}.pt')
            torch.save(test_coeffs, f'{self.save_path}/{self.dataset}/test_coeffs_{str(self.missing_rate)}.pt')
            torch.save(test_labels, f'{self.save_path}/{self.dataset}/test_labels_{str(self.missing_rate)}.pt')

        # Define dataset and dataloder
        train_dataset = TensorDataset(train_data, train_coeffs)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataset = TensorDataset(validation_data, validation_coeffs)
        self.val_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = TensorDataset(test_data, test_coeffs, test_labels)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def data_processor(self, data, data_type):
        length = data.shape[0]
        step = 1
        # sample_num = max(0, (len - self.window_size) // step + 1)
        processed_data = []

        # Process time series data
        for i in range(length):
            if i < data.shape[0] - self.window_size:
                cur_data = data[i * step: i * step + self.window_size]
            else:
                cur_data = data[-self.window_size:]
            processed_data.append(cur_data)
        processed_data = torch.stack(processed_data, dim=0)

        # Process missing data for training data if necessary
        if self.missing_rate > 0.05 and data_type == 'train':
            generator = torch.Generator()
            for Xi in processed_data:
                removed_points = torch.randperm(processed_data.size(1), generator=generator)[
                                 :int(processed_data.size(1) * self.missing_rate)].sort().values
                Xi[removed_points] = float('nan')

        # Calculate the coefficients in NCDE
        ncde_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(processed_data)

        return processed_data, ncde_coeffs

    def normalize(self, data):
        epsilon = 1e-8
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std == 0, epsilon, data_std)
        data = (data - data_mean) / data_std
        return data
