'''
Author: Emy Arts (emy.arts@dlr.de)

Create a dataset from the csv files of flights (either ADS-B or X-plane).
Datasets have the shape n_flights x n_segments x n_features and are normalised according to max and min values (VAL_LIMITS).
Datasets are stored as pytorch files in the folder data/datasets.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from src.pof_functions.weighted_kmeans_optimizer import kmeans_segmentation, alpha_numeric_sort
import os
from sklearn.preprocessing import MinMaxScaler
import random
from typing import List, Union

COLS = ['alt', 'spd', 'roc']


def norm(val: Union[np.ndarray, float], column: str) -> np.ndarray:
    '''
    Min Max scaling with theoretical values
    :param val: value or array of values to be scaled
    :param column: what type of value ('ts', 'alt', 'spd', 'roc')
    :return:
    '''
    limits = {
        ## Altitude and speed taken from A320 and B737 docs
        'ts': [4, 1800],  # artificially set segments to max 30 min
        'alt': [0, 41000],  # ft Both A320 and B737 have same ceiling altitude
        'spd': [0, 470],  # kts Both A320 and B737 have same MMO (max mach operation)
        'roc': [-10000, 10000], # fpm ICAO docs (https://www.icao.int/Meetings/anconf12/Document%20Archive/9863_cons_en.pdf)
    }
    return np.clip((val - limits[column][0]) / (limits[column][1] - limits[column][0]), 0, 1)

def get_name(n_clusters: int, n_features: int, weights: List[float],  kmeans_iters: int, clipping:bool = True) -> str:
    '''
    Function that returns the core of the name of the data set it is creating
    :param n_clusters: int
        as in parameters
    :param n_features: int
        as in parameters
    :param weights: list of float
        as in parameters
    :param clipping: bool
        whether the flights will be clipped (True) or not (False)
    :return: str
        core name of the dataset that can be used to get the desired file
    '''

    if clipping:
        return f"flights_c{n_clusters}_f{n_features}_w{''.join([str(w) for w in weights])}_c_k{kmeans_iters}"
    return f"flights_c{n_clusters}_f{n_features}_w{''.join([str(w) for w in weights])}_k{kmeans_iters}"

def extract_cluster_features(cluster:pd.DataFrame, n_features:int = 7) -> np.ndarray:
    '''
    Extract the features from a segment

    :param cluster: slice of pandas dataframe
        the segment to extract features from
    :param n_features: int
        number of that features represent one cluster
    :param lims: dict
        dictionary with the minimum and maximum value of each column
    :return: numpy array (n_features)
        feature array that represents the segment
    '''

    x = np.zeros(n_features)
    if n_features == 6:
        x[0] = norm(cluster['alt'].dropna().head(1), 'alt')
        x[1] = norm(cluster['alt'].dropna().tail(1), 'alt')
        x[2] = norm(cluster['spd'].dropna().head(1), 'spd')
        x[3] = norm(cluster['spd'].dropna().tail(1), 'spd')
        x[4] = norm(cluster['roc'].dropna().head(1), 'roc')
        x[5] = norm(cluster['roc'].dropna().tail(1), 'roc')
    elif n_features == 7:
        x[0] = norm(len(cluster), 'ts')
        x[1] = norm(cluster['alt'].dropna().head(1), 'alt')
        x[2] = norm(cluster['alt'].dropna().tail(1), 'alt')
        x[3] = norm(cluster['spd'].dropna().head(1), 'spd')
        x[4] = norm(cluster['spd'].dropna().tail(1), 'spd')
        x[5] = norm(cluster['roc'].dropna().head(1), 'roc')
        x[6] = norm(cluster['roc'].dropna().tail(1), 'roc')
    else:
        raise ("Number of features has to be either 6 or 7")
    return x


class Training_data:
    '''Labeled data used to create datasets'''

    def __init__(self, n_clusters: int, weight_vector: List[float], n_features: int, flight_folder: str,
                 ground_truth_folder: str, kmeans_iters: int, clipping=True):
        '''
        Create training dataset from simulation trajectories and their labels

        # :param n_flights: number of flights
        :param n_clusters: number of segments per flight
        :param weight_vector: weights (in range [0, 1]) for K-means segmentation
        :param n_features: number of features that represent a segment
        :param flight_folder: folder with trajectory files
        :param ground_truth_folder: folder with label files
        :param clipping: apply clipping (True) or not (False)
        :param kmeans_iters: number of maximum iterations for flight segmentation
        '''

        flights = []
        labels = []
        self.file_names = []
        for f_idx, file in enumerate(alpha_numeric_sort(os.listdir(ground_truth_folder))):
            l = pd.read_csv(f"{ground_truth_folder}/{file}")
            l['idx'] = f_idx
            labels.append(l)
            f = pd.read_csv(f"{flight_folder}/{file}")
            f['idx'] = f_idx
            f['ground_truth'] = l['phase']
            flights.append(f)
            self.file_names.append(file)
        print("Number of files", f_idx)
        self.n_flights = len(flights)
        self.n_clusters = n_clusters
        self.weights = weight_vector
        self.n_features = n_features
        self.kmeans_iters = kmeans_iters

        if clipping:
            ### Cutting flights to fit better to ADSB:
            #  1) no initial taxi
            #  2) no final taxi
            #  3) no taxi
            #  4) no take off
            #  5) no landing
            #  6) no taxi nor landing

            flights_complete = []
            labels_complete = []
            for j in range(4):
                flights_temp = []
                labels_temp = []
                for i in range(len(flights)):
                    flights_temp.append(flights[i].copy())
                    labels_temp.append(labels[i].copy())
                for i, (f, l) in enumerate(zip(flights_temp, labels_temp)):
                    flights_temp[i]["idx"] = f["idx"] + j * self.n_flights
                    labels_temp[i]["idx"] = l["idx"] + j * self.n_flights
                if j == 0:
                    for i, (fli, lab) in enumerate(zip(flights_temp, labels_temp)):
                        i1 = random.randint(np.where(lab['phase'] == 1)[0][0], np.where(lab['phase'] == 2)[0][-1])
                        labels_temp[i] = lab[i1:].reset_index()
                        flights_temp[i] = fli[i1:].reset_index()
                elif j == 1:
                    for i, (fli, lab) in enumerate(zip(flights_temp, labels_temp)):
                        print(self.file_names[i])
                        i2 = random.randint(np.where(lab['phase'] == 6)[0][0], np.where(lab['phase']==7)[0][-1])
                        labels_temp[i] = lab[:i2].reset_index()
                        flights_temp[i] = fli[:i2].reset_index()
                elif j == 2:
                    for i, (fli, lab) in enumerate(zip(flights_temp, labels_temp)):
                        print(self.file_names[i])
                        i1 = random.randint(np.where(lab['phase']==1)[0][0], np.where(lab['phase'] == 2)[0][-1])
                        i2 = random.randint(np.where(lab['phase'] == 6)[0][0], np.where(lab['phase'] == 7)[0][-1])
                        labels_temp[i] = lab[i1:i2].reset_index()
                        flights_temp[i] = fli[i1:i2].reset_index()
                flights_complete += flights_temp
                labels_complete += labels_temp
                print(len(flights_complete), "Flights")
                self.file_names += self.file_names
            self.data, self.labels, self.flights = self.label_clusters(flights_complete, labels_complete)
            print("Total flights ", pd.concat(self.flights)["idx"].max())
        else:
            self.data, self.labels, self.flights = self.label_clusters(flights, labels)

    def label_clusters(self, flights: List[pd.DataFrame], labels: List[pd.DataFrame]) -> (np.ndarray, np.ndarray, List[pd.DataFrame]):
        '''
        Divide the flight into segments and label them

        :param flights: pandas dataframes with trajectories
        :param labels: pandas dataframes with labels
        :return: (inputs to feed the network, segment labels, trajectories (with segments column))
        '''

        labs = np.zeros((len(flights), self.n_clusters))
        inputs = np.zeros((len(flights), self.n_clusters, self.n_features))
        for flight_n in range(len(flights)):
            print(self.file_names[flight_n])
            label = labels[flight_n]
            flight = flights[flight_n]
            x = [norm(flight[col].copy(), col) for col in flight.columns if col in COLS]
            x = np.array(x).transpose()

            ## K-means
            _, twindows = kmeans_segmentation(x, n_mu=self.n_clusters, weights=self.weights, max_iters=self.kmeans_iters)
            flights[flight_n]['cluster'] = twindows
            counts = pd.value_counts(twindows)
            for i, t in enumerate(twindows):
                if t != twindows[i - 1]:
                    if counts[t] > 0:
                        inputs[flight_n][t] = extract_cluster_features(flight.iloc[i:i + counts[t]], n_features=self.n_features)
                        labs[flight_n][t] = label['phase'][i:(i + counts[t])].value_counts().idxmax()

        return inputs, labs, flights


    def store(self, folder="../data/datasets", val_ratio=0.1):
        '''
        Save the creadted pytorch datasets (training and validation) and the csv overview file

        :param folder: where to store the files
        :param val_ratio: ratio of division of validation and training
        :return:
        '''

        name = get_name(self.n_clusters, self.n_features, self.weights, self.kmeans_iters)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if val_ratio:
            x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=val_ratio)
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            train_ds = TensorDataset(x_train_tensor, y_train_tensor)
            test_ds = TensorDataset(x_test_tensor, y_test_tensor)
            torch.save(train_ds, f"{folder}/train_{name}.pt")
            torch.save(test_ds, f"{folder}/val_{name}.pt")
            flights_df = pd.concat(self.flights)
            flights_df.to_csv(f"{folder}/train_val_csv_{name}.csv")
        else:
            x_tensor = torch.tensor(self.data, dtype=torch.float32)
            y_tensor = torch.tensor(self.labels, dtype=torch.long)
            ds = TensorDataset(x_tensor, y_tensor)
            torch.save(ds, f"{folder}/test_{name}.pt")
            flights_df = pd.concat(self.flights)
            if "index" in flights_df.columns:
                flights_df.drop(columns="index", inplace=True)
            print("Storing", flights_df.idx.max())
            flights_df.to_csv(f"{folder}/test_csv_{name}.csv", index=False)


class Unsupervised_data:
    '''Class that represents unsupervised data used to create a dataset'''
    def __init__(self, n_clusters: int, weight_vector: List[float], n_features: int, kmeans_iters: int, flight_folder:str):
        '''

        :param n_clusters: as in params
        :param weight_vector: as in params
        :param n_features: as in params
        :param kmeans_iters: as in params
        :param flight_folder: folder with flight trajectory csvs
        '''

        flights = []
        self.file_names = []
        for f_idx, file in enumerate(alpha_numeric_sort(os.listdir(flight_folder))):
            f = pd.read_csv(f"{flight_folder}/{file}")
            if not(all(col in f.columns for col in COLS)):
                raise Exception(f"Could not find trajectory columns (alt, spd, roc) in {file}.")
            f['idx'] = f_idx
            flights.append(f)
            self.file_names.append(file)
        self.n_clusters = n_clusters
        self.weights = weight_vector
        self.n_features = n_features
        self.kmeans_iters = kmeans_iters
        self.n_flights = len(flights)
        self.data, self.flights = self.label_clusters(flights)

    def label_clusters(self, flights: List[pd.DataFrame]) -> (np.ndarray, List[pd.DataFrame]):
        '''
        Divide the flight into segments

        :param flights: pandas dataframes with trajectories
        :return: (inputs to be fed to the model, list of pandas dataframes with trajectories with column of segments)
        '''

        inputs = np.zeros((self.n_flights, self.n_clusters, self.n_features))
        for flight_n in range(self.n_flights):
            print(self.file_names[flight_n])
            flight = flights[flight_n]
            x = [norm(flight[col].copy(), col) for col in flight.columns if col in COLS]
            x = np.array(x).transpose()

            ## K-means
            _, twindows = kmeans_segmentation(x, n_mu=self.n_clusters, weights=self.weights, max_iters=self.kmeans_iters)
            flights[flight_n]['cluster'] = twindows
            counts = pd.value_counts(twindows)
            for i, t in enumerate(twindows):
                if t != twindows[i - 1]:
                    inputs[flight_n][t] = extract_cluster_features(flight.loc[i:i + counts[t]], n_features=self.n_features)
        return inputs, flights


    def store(self, folder:str = "unsupervised_datasets"):
        '''
        Store unsupervised test pytorch dataset and overview csv

        :param folder: where to save the stored files
        :return:
        '''
        if not os.path.exists(folder):
            os.mkdir(folder)
        x = torch.tensor(self.data, dtype=torch.float32)
        ds = TensorDataset(x)
        name = get_name(self.n_clusters, self.n_features, self.weights, self.kmeans_iters)
        torch.save(ds, f"{folder}/unsupervised_{name}.pt")
        flights_df = pd.concat(self.flights)
        flights_df.to_csv(f"{folder}/unsupervised_csv_{name}.csv")


if __name__ == '__main__':

    while "src" in os.getcwd():
        os.chdir("..")
        print(f"Changed directory to {os.getcwd()}")

    supervised = False

    params = {
        "n_clusters": 160,
        "n_features": 7,
        "batch_size": 16,
        "weights": [0.7, 0.8, 0.1],
        "n_layers": 2,
        "learning_rate": 0.01,
        "hidden_dim": 16,
        "n_epochs": 3500,
        "alpha": 0,
        "trial": 2,
        "kmeans_iters": 100
    }

    dataset_name = get_name(params["n_clusters"], params["n_features"], params["weights"], clipping=True, kmeans_iters=params["kmeans_iters"])
    if not os.path.exists(f"/data/datasets/train_{dataset_name}.pt"):
        print(f"Creating dataset {dataset_name}")
        if supervised:
            print("Supervised data")
            data_creator = Training_data(n_clusters=params['n_clusters'],
                                         weight_vector=params['weights'],
                                         n_features=params['n_features'],
                                         kmeans_iters=params['kmeans_iters'],
                                         # flight_folder=f"data/preprocessed/trajectories_train",
                                         # ground_truth_folder=f"data/preprocessed/labels_train",
                                         # clipping = True)
                                         flight_folder=f"data/preprocessed/trajectories_test",
                                         ground_truth_folder=f"data/preprocessed/labels_test",
                                         clipping=True)
            data_creator.store(f"data/datasets", val_ratio=None)
        else:
            print("Unsupervised data")
            print(os.getcwd())
            data_creator = Unsupervised_data(n_clusters=params["n_clusters"],
                                         weight_vector=params["weights"],
                                         n_features=params['n_features'],
                                        kmeans_iters=params['kmeans_iters'],
                                         flight_folder=f"data/ADSB/preprocessed/csvs")
            data_creator.store(f"data/datasets")