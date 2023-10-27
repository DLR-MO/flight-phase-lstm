# SPDX-FileCopyrightText: 2023 German Aerospace Center
#
# SPDX-License-Identifier: MIT

'''
Author: Emy Arts (emy.arts@dlr.de)

Evaluation on the test dataset specified of a network trained with network.py prints accuracy of classification.
Stores classification images
'''

import re
from pof_functions.create_dataset import get_name, extract_cluster_features, Unsupervised_data
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
import random
from pof_functions.network import POF_net
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# np.testing.suppress_warnings(forwarding_rule='always')
number_to_label = {
    0: "taxi",
    1: "take-off",
    2: "initial climb",
    3: "climb",
    4: "cruise",
    5: "descent",
    6: "approach",
    7: "landing"
}

colormap = {"taxi": "#000000",
            "take-off": "#66FF33",
            "initial climb": "#10CC00",
            "climb": "#548235",
            "cruise": "#FFCC05",
            "descent": "#0533CC",
            "approach": "#00B0F0",
            "landing": "#00FFFF",
            }


def get_legendlines(colors):
    lines = []
    if colors[0] in colormap.values():
        cm = colormap
    else:
        cm = colormap
    for lab, col in cm.items():
        if col in colors:
            lines.append(Line2D([0], [0], color=col, label=lab))
    return lines


class eval():

    def __init__(self, pof_model: POF_net, models_folder: str = "../results/models", test=True, unsupervised=False):
        '''

        :param pof_model: network to evaluate
        :param models_folder: folder where the saved models can be found
        :param val: number of flights trained with if the evaluation set is not the same as the training set
        '''

        # The model to be evaluated and its parameters
        self.model = pof_model.get_model()
        self.params = pof_model.get_config_dict()
        self.model_name = pof_model.get_name()

        # update the weights of the model with highest accuracy weights during training
        self.model.load_state_dict(torch.load(f"{models_folder}/{self.model_name}.pt", map_location=torch.device('cpu')))
        dataset = get_name(self.params["n_clusters"], self.params["n_features"], self.params["weights"], self.params["kmeans_iters"])

        # Store dataframe that contains all the flights of the dataset to evaluate on
        if unsupervised:
            self.data = pd.read_csv(f"data/datasets/unsupervised_csv_{dataset}.csv")
        elif test:
            self.data = pd.read_csv(f"data/datasets/test_csv_{dataset}.csv").dropna(axis=0)
        else:
            self.data = pd.read_csv(f"data/datasets/train_val_csv_{dataset}.csv").dropna(axis=0)
        # if test:
        #     self.data = pd.read_csv(f"data/datasets/temp/test_csv_{dataset}.csv").dropna(axis=0)
        # else:
        #     self.data = pd.read_csv(f"data/datasets/temp/train_val_csv_{dataset}.csv").dropna(axis=0)
        print("Max index: ", self.data["idx"].max())

    def set_prediction_columns(self) -> pd.DataFrame:
        '''
        Uses the model to find the identification labels and puts them in the "prediction" column
        Store the dataframe with identification labels to be used for comparison.

        :return: overview dataframe with all the flights
        '''

        flights = self.data.copy().groupby(by=["idx", "cluster"])
        inputs = np.zeros((self.data["idx"].max()+1, self.params["n_clusters"], self.params["n_features"]))
        for idx, t in flights:
            inputs[int(idx[0])][int(idx[1])] = extract_cluster_features(t, n_features=self.params['n_features'])
        with torch.no_grad():
            input_torch = torch.tensor(inputs, dtype=torch.float32)
            preds = self.output_prediction(input_torch)
        self.data["prediction"] = self.data.apply(lambda row: preds[int(row["idx"])][int(row["cluster"])], axis=1)
        return self.data

    def output_prediction(self, inps: torch.Tensor, force_transitions=True) -> np.ndarray:
        '''
        From the input interpret the output of the LSTM into labels

        :param inps: the inputs to feed the model
        :param force_transitions: most probable labels maintaining correct transitions(True) or argmax (False)
        :return: the labels for each segment of each flight
        '''

        preds = np.zeros((self.data["idx"].max()+1, self.params["n_clusters"], len(number_to_label)))
        for i in range(self.data["idx"].max()+1):
            preds[i] = self.model(inps[i]).numpy()
        max_labs = np.nanargmax(preds[:, :, :], axis=2)
        if force_transitions:
            state_labs = np.zeros_like(max_labs)
            state_transitions = re.compile("0*1*2*3*4*5*6*7*0*")
            for i, lab in enumerate(max_labs):
                if state_transitions.match(np.array2string(lab, separator='', max_line_width=self.params['n_clusters']+2)[1:-1]).end() == self.params['n_clusters']:
                    state_labs[i] = lab
                else:
                    print("Found invalid transition in flight ", i)
                    for iteration in range(100):
                        if state_transitions.match(np.array2string(max_labs[i], separator='', max_line_width=self.params['n_clusters'] + 2)[1:-1]).end() >= self.params['n_clusters']:
                            break
                        blocks = lab[abs(np.diff(lab, prepend=0)) > 0]
                        block_idx = np.append(np.argwhere(abs(np.diff(lab, prepend=0)) > 0).flatten(), self.params['n_clusters'])
                        taxis = np.argwhere(blocks == 1).flatten()

                        # Taxi has to be handled separately since it occurs twice
                        taxi_adjustment = False
                        for t in taxis:
                            if t == 1:
                                p_aac = np.mean(preds[i, :block_idx[t], 1]) # setting first block to taxi
                                p_acc = np.mean(preds[i, block_idx[t]:block_idx[t+1], blocks[0]]) # setting wrong taxi block to first
                                if p_aac < p_acc:
                                    max_labs[i, block_idx[t]:block_idx[t+1]] = blocks[0]
                                else:
                                    max_labs[i, :block_idx[t]] = 1
                                taxi_adjustment = True
                            elif t == len(blocks)-2:
                                p_aac = np.mean(preds[i, block_idx[t]:block_idx[t+1], blocks[t+1]])  # setting first block to taxi
                                p_acc = np.mean(preds[i, block_idx[t+1]:, 1])  # setting wrong taxi block to first
                                if p_aac < p_acc:
                                    max_labs[i, block_idx[t+1]:] = 1
                                else:
                                    max_labs[i, block_idx[t]:block_idx[t+1]] = blocks[t+1]
                                taxi_adjustment = True
                        if taxi_adjustment == True: continue
                        prob_block = state_transitions.match(np.array2string(blocks, separator='', max_line_width=self.params['n_clusters']+2)[1:-1]).end()-1
                        if blocks[prob_block] == blocks[prob_block-2]:
                            # On jump to a smaller number (a previous phase) the wrong one is excluded in last valid progression
                            prob_block -= 1
                        p_aac = np.mean(preds[i, block_idx[prob_block]:block_idx[prob_block+1], blocks[prob_block-1]])
                        p_acc = np.mean(preds[i, block_idx[prob_block]:block_idx[prob_block+1], blocks[prob_block+1]])
                        p_abb = np.mean(preds[i, block_idx[prob_block+1]:block_idx[prob_block+2], blocks[prob_block]])
                        max_prob = max(p_aac, p_acc, p_abb)
                        if p_aac == max_prob:
                            max_labs[i, block_idx[prob_block]:block_idx[prob_block + 1]] = blocks[prob_block - 1]
                        elif p_acc == max_prob:
                            max_labs[i, block_idx[prob_block]:block_idx[prob_block + 1]] = blocks[prob_block + 1]
                        elif p_abb == max_prob:
                            max_labs[i, block_idx[prob_block + 1]:block_idx[prob_block + 2]] = blocks[prob_block]
                        else:
                            print("No max")
                            raise ValueError
        return max_labs

    def get_accuracy(self, col="prediction")->np.ndarray:
        '''
        Print and return the accuracy of the identification

        :param col: name of the colum with identification values
        :return:  accuracy of all flights
        '''

        if not col in self.data.columns:
            self.set_prediction_columns()
        acc = (self.data["ground_truth"] == self.data[col]).sum()/len(self.data)
        acc_per_flight = np.array([(self.data[self.data['idx'] == flight_n]["ground_truth"] ==
                                    self.data[self.data['idx'] == flight_n][col]).sum() / len(
            self.data[self.data['idx'] == flight_n]) for flight_n in range(self.data["idx"].max())])
        print(f"Overall accuracy {round(acc*100, 2)}%")
        n_classes = max(number_to_label.keys()) + 1
        print("Accuracy for each phase:")
        print("Recall:")
        phase_counts = self.data["ground_truth"].value_counts().to_dict()
        acc_r = np.zeros(n_classes) # Number of phases
        for phase, count in phase_counts.items():
            phase_data = self.data[self.data["ground_truth"] == phase]
            acc_r[int(phase)] = (phase_data[col] == phase).sum()/count
            print(f"\t{number_to_label[int(phase)]}({int(count/(self.data['idx'].max() +1))}s) {round(acc_r[int(phase)]*100, 2)}%")
        print("Precision:")
        phase_counts = self.data[col].value_counts().to_dict()
        acc_p = np.zeros(n_classes)
        for phase, count in phase_counts.items():
            phase_data = self.data[self.data[col] == phase]
            acc_p[int(phase)] = (phase_data[col] == phase_data["ground_truth"]).sum()/count
            print(f"\t{number_to_label[int(phase)]}({int(count/(self.data['idx'].max()+1))}s) {round(acc_p[int(phase)]*100, 2)}%")
        print(f"Average phase recall {round(acc_r[1:].mean() * 100, 2)}%")
        print(f"Average phase precision {round(acc_p[1:].mean() * 100, 2)}%")
        return acc_per_flight

    def get_val_accuracy(self, test = False):
        '''
        Print accuracy per segment rather than second

        :param test: test set (False) or validation set (True)
        '''

        if test:
            testing_set = torch.load(f"data/datasets/test_{get_name(params['n_clusters'], params['n_features'], params['weights'], params['kmeans_iters'])}.pt")
        else:
            testing_set = torch.load(f"data/datasets/val_{get_name(params['n_clusters'], params['n_features'], params['weights'], params['kmeans_iters'])}.pt")
        test_data = DataLoader(testing_set, shuffle=True)
        acc_test = np.zeros(len(test_data))
        precision = np.empty((len(test_data), len(number_to_label)))
        recall = np.empty((len(test_data), len(number_to_label)))
        print(len(test_data))
        with torch.no_grad():
            for i, (x, y) in enumerate(test_data):
                x = x[0]
                y = y[0]
                out = self.model(x)
                acc_test[i] = (out.argmax(axis=1) == y).sum()/self.params["n_clusters"]
                values = torch.bincount(y)
                out_argmax = out.argmax(axis=1)
                slice = y[torch.where(out_argmax == y)[0]]
                recall[i] = [((slice == phase).sum() / values[phase]).numpy() if (phase < len(values)) and (values[phase] > 0) else np.nan for phase in range(len(number_to_label))]
                values = torch.bincount(out_argmax)
                slice = out_argmax[torch.where(out_argmax == y)[0]]
                precision[i] = [((slice == phase).sum() / values[phase]).numpy() if (phase < len(values)) and (values[phase] > 0) else np.nan for phase in range(len(number_to_label))]
        print("Precision number of nans: ", np.count_nonzero(np.isnan(precision)))
        print("Recall number of nans: ", np.count_nonzero(np.isnan(recall)))
        print(f"Average test accuracy {round(acc_test.mean()*100, 2)}%")
        print(f"Test accuracy per phase")
        for i in range(len(number_to_label)):
            print(f"\t{number_to_label[i]} \tprecision {np.round(np.nanmean(precision, axis=0)[i] * 100, 2)}% \n\t\t\t recall {np.round(np.nanmean(recall, axis=0)[i] * 100, 2)}%")
        print(f"Average test accuracy related to phases {np.round(np.nanmean((precision + recall)/2) * 100, 2)}%")
        return acc_test

    def store_predictions(self, name):
        for col in ['Unnamed: 0', 'level_0', 'index']:
            if col in self.data.columns:
                self.data.drop(columns=col, inplace=True)
        self.data['prediction'] = [number_to_label[p] for p in self.data['prediction']]
        if not os.path.exists("results/predicted_labels/"):
            os.mkdir("results/predicted_labels/")
        self.data.to_csv(f"results/predicted_labels/{name}.csv", index=False)

    def plot_prediction(self, flight_number:int=None, n_samples:int=None, with_truth=True, save=False):
        '''
        Show the models identified labels

        :param flight_number: int, the number of the single flight to plot
        :param n_samples: int, the number of randomly taken flights to plot
        :param with_truth: boolean, show the correct and incorrect labels in the background (True) or not (False)
        :param save: boolean, store the figure (True) or save it (False)

        either flight_number or n_samples has to be given
        save stores the files in the "figs" folder and creates it if not present.
        '''
        plt.rcParams.update({'font.size': 20})
        plt.rc('legend', fontsize=24)
        if not "prediction" in self.data.columns:
            self.set_prediction_columns()
        if not flight_number is None:
            flights = [self.data[self.data["idx"] == flight_number]]
        elif not n_samples is None:
            sample = random.sample(range(self.data["idx"].max()), n_samples)
            flights = [self.data[self.data["idx"] == i] for i in sample]
        else:
            raise Exception("Need either flight number or number of samples")
        for flight in flights:
            colors_c = [colormap[number_to_label[l]] for l in flight['prediction']]

            flight["ts"] = flight["ts"].astype(int)
            flight["ts"] = (flight["ts"] - np.full(flight.shape[0], flight["ts"].head(1)))

            h_line_times = flight["ts"][abs(flight["cluster"].diff(1)) > 0]
            if with_truth:
                fig, ax = plt.subplots(5, figsize=(18, 12), sharex=True, gridspec_kw={'hspace':0, 'height_ratios': [3, 1.3, 1.3, 0.22, 0.22]})
                ax[0].scatter(flight['ts'], flight['alt'], s=5, c=colors_c, zorder=10)
                ax[0].set_ylim(0)
                ax[1].scatter(flight['ts'], flight['spd'], s=5, c=colors_c, zorder=10)
                ax[1].set_ylim(0)
                ax[2].scatter(flight['ts'], flight['roc'], s=5, c=colors_c, zorder=10)

                # Vertical cluster border lines
                for t in h_line_times:
                    ax[0].axvline(t, color='gray', linestyle='--', linewidth=0.5, zorder=5)
                    ax[1].axvline(t, color='gray', linestyle='--', linewidth=0.5, zorder=5)
                    ax[2].axvline(t, color='gray', linestyle='--', linewidth=0.5, zorder=5)

                # Main parameters
                acc = (flight['prediction'] == flight['ground_truth']).sum()/len(flight)
                ax[0].legend(handles=get_legendlines(colors_c), prop={'size': 18}, fontsize=30)
                ax[0].set_ylabel("altitude (ft)", labelpad=20)
                ax[1].set_ylabel("speed (kts)", labelpad=40)
                ax[2].set_ylabel("RoC (fpm)", labelpad=10)
                ax[3].yaxis.set_visible(False)
                ax[4].yaxis.set_visible(False)
                plt.suptitle(f"{round(acc *100, 2)}% accuracy")
                plt.xlabel("time (s)")
                time_10 = flight['ts'].max() / 10
                plt.xlim([0 - time_10 / 10, flight['ts'].max() + time_10 / 10])

                # Labels comparison truth vs prediction
                ts = flight["ts"].iloc[0].copy()
                for i, row in flight.iterrows():
                    ax[3].axvspan(ts, row["ts"], facecolor=colormap[number_to_label[row["prediction"]]])
                    ax[4].axvspan(ts, row["ts"], facecolor=colormap[number_to_label[row["ground_truth"]]])
                    ts = row["ts"]
                ax[3].text(flight['ts'].max() / 2 - time_10 / 4, 0.12, 'prediction', color='white', fontsize=18)
                ax[4].text(flight['ts'].max() / 2 - time_10 / 4, 0.12, 'truth', color='white', fontsize=18)

                if save:
                    if not os.path.exists("figs"):
                        os.mkdir("figs")
                    plt.savefig(f"figs/{self.model_name}_{flight['idx'].values[0]}.png")
                    plt.clf()
                else:
                    plt.show()
            else:
                fig, ax = plt.subplots(3, figsize=(16, 9), sharex=True, gridspec_kw={'hspace':0, 'height_ratios': [3, 1.2, 1.2]})
                ax[0].scatter(flight['ts'], flight['alt'], s=5, c=colors_c, zorder=10)
                ax[0].set_ylim(0)
                ax[1].scatter(flight['ts'], flight['spd'], s=5, c=colors_c, zorder=10)
                ax[1].set_ylim((0, flight['spd'].max()*1.5))
                ax[2].scatter(flight['ts'], flight['roc'], s=5, c=colors_c, zorder=10)
                roc_lim = max(abs(flight['roc'].min()), abs(flight['roc'].max()))*2
                ax[2].set_ylim((-roc_lim, roc_lim))
                ax[0].set_ylabel("altitude (ft)", labelpad=20)
                ax[1].set_ylabel("speed (kts)", labelpad=40)
                ax[2].set_ylabel("RoC (fpm)", labelpad=10)
                plt.xlabel("time (s)")
                ax[0].legend(handles=get_legendlines(colors_c), prop={'size': 18}, fontsize=30)
                for t in h_line_times:
                    ax[0].axvline(t, color='gray', linestyle='--', linewidth=0.5, zorder=5)
                    ax[1].axvline(t, color='gray', linestyle='--', linewidth=0.5, zorder=5)
                    ax[2].axvline(t, color='gray', linestyle='--', linewidth=0.5, zorder=5)
                if save:
                    if not os.path.exists("results/figs"):
                        os.mkdir("results/figs")
                    plt.savefig(f"results/figs/{self.model_name}_{flight['idx'].values[0]}.png")
                    plt.clf()
                else:
                    plt.show()

    def get_flight_stats(self):
        print(self.data.groupby(by='idx').mean())


if __name__ == '__main__':

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

    while "src" in os.getcwd():
        os.chdir("..")
        print(f"Changed directory to {os.getcwd()}")

    parser = argparse.ArgumentParser(description="Evaluation of a pretrained model")
    parser.add_argument("--custom_data_path", type=str, default=None, help="path to the folder containing custom OpenSkynet files")
    args = parser.parse_args()
    custom_data_path = args.custom_data_path

    if custom_data_path is not None:
        try:
            list_of_flights = os.listdir(custom_data_path)
        except:
            raise Exception("The custem data folder is not found")
        params['n_flights'] = len(list_of_flights)
        print("Creating dataset")
        data_creator = Unsupervised_data(n_clusters=params["n_clusters"],
                                         weight_vector=params['weights'],
                                         n_features=params['n_features'],
                                         kmeans_iters=params['kmeans_iters'],
                                         flight_folder=custom_data_path)
        data_creator.store(f"data/datasets")


    net = POF_net(params, dev="cpu")
    if custom_data_path is None:
        eval = eval(net, models_folder="./results/models", test=True)
        acc = eval.get_accuracy()
        acc_val = eval.get_val_accuracy(True)
        eval.plot_prediction(n_samples=1, with_truth=True)
    else:
        eval = eval(net, models_folder="./results/models", test=True, unsupervised=True)
        for i in range(params['n_flights']):
            eval.plot_prediction(flight_number=i, save=True, with_truth=False)
        eval.store_predictions("custom_dataset")

