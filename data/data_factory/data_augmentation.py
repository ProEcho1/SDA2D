from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
import numpy as np
import pandas as pd
import random
import scipy.io
import os
import mat73
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.mixture import GaussianMixture
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE
from colorama import Style, Fore, init

init(autoreset=True)

from customized_utils import Utils


class DataGenerator():
    def __init__(self, generate_duplicates=False, n_samples_threshold=1000, n_samples_up_threshold=10000,
                 show_statistic=False, batch_size=32):
        """
        Only global parameters should be provided in the DataGenerator instantiation

        seed: seed for reproducible experimental results
        dataset: dataset name
        test_size: testing data size
        """

        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold
        self.n_samples_up_threshold = n_samples_up_threshold
        self.show_statistic = show_statistic
        self.batch_size = batch_size
        self.gm = None
        self.copula = None
        self.utils = Utils()

    def fit_best_distribution_sampler(self, X, y):
        # Only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        # Select the best n_components based on the BIC value
        metric_list = []
        n_components_list = list(np.arange(1, 10))
        for n_components in n_components_list:
            gm = GaussianMixture(n_components=n_components).fit(X)
            metric_list.append(gm.bic(X))
        best_n_components = n_components_list[np.argmin(metric_list)]
        # Refit based on the best n_components
        self.gm = GaussianMixture(n_components=best_n_components).fit(X)
        print(Fore.RED + f'GaussianMixture initialization finished.')

        # self.copula = VineCopula('center')  # Default is the C-vine copula
        # self.copula.fit(pd.DataFrame(X))
        # print(Fore.RED + 'Copula initialization finished.')

    def generate_realistic_synthetic(self, X, y, realistic_synthetic_mode, alpha, percentage, anomaly_ratio):
        """
        Currently, four types of realistic synthetic outliers can be generated:
        1. local outliers: where normal data follows the GMM distribution, and anomalies follow the GMM distribution with modified covariance
        2. global outliers: where normal data follows the GMM distribution, and anomalies follow the uniform distribution
        3. dependency outliers: where normal data follows the vine copula distribution, and anomalies follow the independent distribution captured by GaussianKDE
        4. cluster outliers: where normal data follows the GMM distribution, and anomalies follow the GMM distribution with modified mean

        :param X: input X
        :param y: input y
        :param realistic_synthetic_mode: the type of generated outliers
        :param alpha: the scaling parameter for controlling the generated local and cluster anomalies
        :param percentage: controlling the generated global anomalies
        """

        if realistic_synthetic_mode in ['local', 'cluster', 'dependency', 'global']:
            pass
        else:
            raise NotImplementedError

        # The number of normal data and anomalies
        pts_n = len(np.where(y == 0)[0])
        pts_a = int(pts_n * anomaly_ratio)

        # Only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        # Generate the synthetic normal data
        if realistic_synthetic_mode in ['local', 'cluster', 'global']:
            # Generate the synthetic normal data
            X_synthetic_normal = self.gm.sample(pts_n)[0]
        elif realistic_synthetic_mode == 'dependency':
            # Sample to generate synthetic normal data
            X_synthetic_normal = self.copula.sample(pts_n).values

        gm = deepcopy(self.gm)

        # Generate the synthetic abnormal data
        if realistic_synthetic_mode == 'local':
            # Generate the synthetic anomalies (local outliers)
            gm.covariances_ = alpha * gm.covariances_
            X_synthetic_anomalies = gm.sample(pts_a)[0]
        elif realistic_synthetic_mode == 'cluster':
            # Generate the clustering synthetic anomalies
            gm.means_ = alpha * gm.means_
            X_synthetic_anomalies = gm.sample(pts_a)[0]
        elif realistic_synthetic_mode == 'dependency':
            X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))
            # Using the GuassianKDE for generating independent feature
            for i in range(X.shape[1]):
                kde = GaussianKDE()
                kde.fit(X[:, i])
                X_synthetic_anomalies[:, i] = kde.sample(pts_a)
        elif realistic_synthetic_mode == 'global':
            # Generate the synthetic anomalies (global outliers)
            X_synthetic_anomalies = []
            for i in range(X_synthetic_normal.shape[1]):
                low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
                high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)
                X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))
            X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

        X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
        y = np.append(np.repeat(0, X_synthetic_normal.shape[0]),
                      np.repeat(1, X_synthetic_anomalies.shape[0]))

        print(Fore.YELLOW + f'Normal points: {len(X_synthetic_normal)}, abnormal points: {len(X_synthetic_anomalies)}')

        return X, y

    def generator(self, dataset='SMD', alpha=5, percentage=0.1):
        basic_path = f'../datasets/{dataset.upper()}'
        if dataset == 'SMD':
            train_data = np.load(basic_path + "/SMD_train.npy")[:, :]
            test_data = np.load(basic_path + "/SMD_test.npy")[:, :]
            test_labels = np.load(basic_path + "/SMD_test_label.npy")[:]
        elif dataset == 'MSL':
            train_data = np.load(basic_path + "/MSL_train.npy")
            test_data = np.load(basic_path + "/MSL_test.npy")
            test_labels = np.load(basic_path + "/MSL_test_label.npy")
        elif dataset == 'PSM':
            data = pd.read_csv(basic_path + '/train.csv')
            data = data.values[:, 1:]
            train_data = np.nan_to_num(data)
            data = pd.read_csv(basic_path + '/test.csv')
            data = data.values[:, 1:]
            test_data = np.nan_to_num(data)
            test_labels = pd.read_csv(basic_path + '/test_label.csv').values[:, 1:]
        elif dataset == 'SWaT':
            train_data = pd.read_csv(os.path.join(basic_path, 'swat_train2.csv'))
            test_data = pd.read_csv(os.path.join(basic_path, 'swat2.csv'))
            test_data_copy = deepcopy(test_data)
            train_data = train_data.values[:, :-1]
            test_data = test_data.values[:, :-1]
            test_labels = test_data_copy.values[:, -1:]
        elif dataset == 'SMAP':
            train_data = np.load(basic_path + "/SMAP_train.npy")
            test_data = np.load(basic_path + "/SMAP_test.npy")
            test_labels = np.load(basic_path + "/SMAP_test_label.npy")

        X = train_data
        y = np.zeros((X.shape[0],))
        X_test = test_data
        y_test = test_labels

        # Generate realistic synthetic outliers
        if not os.path.exists('../datasets/synthetic'):
            os.makedirs('../datasets/synthetic')

        # Initialize GaussianMixture and Copula
        self.fit_best_distribution_sampler(X, y)

        # Generate augmented datasets
        # Ignore 'dependency' mode for intensive calculation
        for realistic_synthetic_mode in ['local', 'global', 'cluster']:
            for anomaly_ratio in [0.1, 0.2, 0.3]:
                X_test = test_data
                y_test = test_labels
                print(
                    Fore.RED + f'Processing {dataset} with {realistic_synthetic_mode} under anomaly ratio {anomaly_ratio}....')
                X, y = self.generate_realistic_synthetic(X, y, realistic_synthetic_mode=realistic_synthetic_mode,
                                                         alpha=alpha, percentage=percentage,
                                                         anomaly_ratio=anomaly_ratio)

                # Show the statistic
                self.utils.data_description(X=X, y=y)

                # Resample for balanced distribution in a single batch
                X, y = self.utils.sampler(X, y, self.batch_size)

                # Split the whole data into training dataset and validation dataset
                X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

                # Normalization
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_validation = scaler.transform(X_validation)
                X_test = scaler.transform(X_test)

                # Transfer data from Numpy() to Tensor()
                X_train = torch.from_numpy(X_train).float()
                X_validation = torch.from_numpy(X_validation).float()
                X_test = torch.from_numpy(X_test).float()
                y_train = torch.from_numpy(y_train).long()
                y_validation = torch.from_numpy(y_validation).long()
                y_test = torch.from_numpy(y_test).long()

                # # Get coefficients for NCDE
                # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
                # validation_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_validation)
                # test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test)

                # Save data and coeffs
                os.makedirs(f'../datasets/synthetic/{dataset}/', exist_ok=True)
                torch.save(X_train,
                           f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_train_data.pt')
                torch.save(y_train,
                           f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_train_labels.pt')
                # torch.save(train_coeffs,
                #            f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_train_coeffs.pt')
                torch.save(X_validation,
                           f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_validation_data.pt')
                torch.save(y_validation,
                           f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_validation_labels.pt')
                # torch.save(validation_coeffs,
                #            f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_validation_coeffs.pt')
                torch.save(X_test,
                           f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_test_data.pt')
                torch.save(y_test,
                           f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_test_labels.pt')
                # torch.save(test_coeffs,
                #            f'../datasets/synthetic/{dataset}/{realistic_synthetic_mode}_{str(anomaly_ratio)}_test_coeffs.pt')

                # Finish report
                print(
                    Fore.RED + f'Process {dataset} with anomaly ratio {anomaly_ratio} under mode {realistic_synthetic_mode} finished.')


if __name__ == '__main__':
    # Define seeding for reproducibility
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_generator = DataGenerator()

    # Start data augmentation
    for dataset in ['SMD', 'MSL', 'PSM', 'SWaT', 'SMAP']:
        data_generator.generator(dataset=dataset, alpha=5, percentage=0.1)
