'''
Author: Emy Arts (emy.arts@dlr.de)

Training a LSTM network with a dataset created through create_dataset.py
Training statistics are stored in results/csv and the model in results/models as a pytorch file.
'''

import torch.optim as optim
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
from pof_functions.create_dataset import Training_data, get_name
from datetime import datetime
import copy

def get_full_name(config_dict:dict)->str:
	'''
	:param config_dict: dictionary with parameters
	:return: core name of the model
	'''
	return f"net_c{config_dict['n_clusters']}_f{config_dict['n_features']}_w{''.join([str(w) for w in config_dict['weights']])}_hu{config_dict['hidden_dim']}_" \
		   f"d{config_dict['n_layers']}_lr{config_dict['learning_rate']}_bs{config_dict['batch_size']}_a{config_dict['alpha']}_k{config_dict['kmeans_iters']}_t{config_dict['trial']}"

dtype = torch.double
# device = torch.device("cpu")
device = torch.device("cuda")

class LSTMTagger(nn.Module):
	'''
	Core classification LSTM
	slight adjustments made from source: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
	'''
	def __init__(self, n_features: int, hidden_dim: int, n_classes: int, n_layers: int):
		torch.backends.cudnn.enabled = False
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers)

		self.hidden2tag = nn.Linear(hidden_dim, n_classes)

	def forward(self, x):
		lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
		tag_space = self.hidden2tag(lstm_out.view(len(x), -1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores

class POF_net():

	def __init__(self, config_dict: dict, dev="cuda"):
		'''

		:param config_dict: dictionary with model parameters
		:param dev: device on which to train ("cuda"->gpu, "cpu"->cpu)
		'''
		self.N_FEATURES = config_dict['n_features']
		self.N_CLUSTERS = config_dict['n_clusters']
		self.N_CLASSES = 8
		self.LEARNING_RATE = config_dict['learning_rate']
		self.HIDDEN_DIM = config_dict['hidden_dim']
		self.N_LAYERS = config_dict['n_layers']
		self.BATCH = config_dict["batch_size"]
		self.alpha = config_dict['alpha'] # penalty influence factor
		self.device = torch.device(dev)
		self.model = LSTMTagger(self.N_FEATURES, self.HIDDEN_DIM, self.N_CLASSES, self.N_LAYERS)
		self.model.to(self.device)
		self.loss_function = nn.NLLLoss()
		self.optimizer = optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE)
		self.logger = logging.getLogger("Net logger")
		self.name = get_full_name(config_dict)
		self.config_dict = config_dict
		print("Creating log file(s)")
		fh = logging.FileHandler(f"results/logs/{self.name}.log")
		sh = logging.StreamHandler()
		self.logger.setLevel(logging.INFO)
		self.logger.addHandler(fh)
		self.logger.addHandler(sh)
		self.logger.info(config_dict)
		if os.path.exists('../.remote_folder'):
			with open('../.remote_folder', 'r') as stream:
				self.remote_store = stream.read()
			self.logger.info("Remote monitoring and storage active.")
			self.logger_remote = logging.getLogger("Remote logger")
			if not os.path.exists(f"{self.remote_store}/logs"):
				os.makedirs(f"{self.remote_store}/logs")
			if not os.path.exists(f"{self.remote_store}/models"):
				os.makedirs(f"{self.remote_store}/models")
			if not os.path.exists(f"{self.remote_store}/csvs"):
				os.makedirs(f"{self.remote_store}/csvs")
			net_handler = logging.FileHandler(f"{self.remote_store}/logs/{self.name}.log")
			self.logger_remote.addHandler(net_handler)
			self.logger_remote.setLevel(logging.INFO)
			self.logger_remote.info(config_dict)
		else:
			self.remote_store = None

	def get_model(self) -> LSTMTagger: return self.model

	def set_model(self, model: LSTMTagger): self.model = model

	def get_name(self) -> str: return self.name

	def get_config_dict(self) -> str: return self.config_dict


	def train(self, n_epochs:int, training:str, validation:str)->np.ndarray:
		'''
		Training the LSTM

		:param n_epochs: number of epochs
		:param training: training dataset pytroch file
		:param validation: validation dataset pytroch file
		:return: average validation accuracy for each epoch
		'''

		training_set = torch.load(training)
		validation_set = torch.load(validation)
		training_data = DataLoader(training_set, shuffle=True, batch_size=self.BATCH)
		validation_data = DataLoader(validation_set, shuffle=True)
		self.logger.info("Starting training")
		loss_array = np.zeros((n_epochs, len(training_data)))
		acc_train = np.zeros((n_epochs, len(training_data)))
		acc_val = np.zeros((n_epochs, len(validation_data)))
		c_acc_train = np.zeros((n_epochs, len(training_data)))
		c_acc_val = np.zeros((n_epochs, len(validation_data)))
		for epoch in range(n_epochs):

			# Training
			for i, (x, y) in enumerate(training_data):
				x = x[0].to(self.device)
				y = y[0].to(self.device)
				self.model.zero_grad()
				out = self.model(x)
				loss = self.loss_function(out, y)

				# Computation of the loss penalty
				if self.alpha > 0:
					out_argmax = out.argmax(axis=1)
					values = torch.bincount(y)
					slice = y[torch.where(out_argmax != y)[0]]
					false_neg = [((slice == phase).sum() / values[phase]) for phase in range(self.N_CLASSES) if
								 (phase < len(values)) and (values[phase] > 0)]
					values = torch.bincount(out_argmax)
					slice = out_argmax[torch.where(out_argmax != y)[0]]
					false_pos = [((slice == phase).sum() / values[phase]) for phase in range(self.N_CLASSES) if
								 (phase < len(values)) and values[phase] > 0]
					avg_false = (torch.sum((torch.stack(false_pos))) + torch.sum(torch.stack(false_neg))) / (len(false_pos) + len(false_neg))
					c_acc_train[epoch][i] = 1 - avg_false
					loss += self.alpha * avg_false

				loss_array[epoch][i] = loss
				acc_train[epoch][i] = (out.argmax(axis=1) == y).sum()/self.N_CLUSTERS
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
				self.optimizer.step()

			# Validation
			with torch.no_grad():
				for i, (x, y) in enumerate(validation_data):
					x = x[0].to(self.device)
					y = y[0].to(self.device)
					out = self.model(x)
					out_argmax = out.argmax(axis=1)
					acc_val[epoch][i] = (out_argmax == y).sum() / self.N_CLUSTERS
					values = torch.bincount(y)
					slice = y[torch.where(out_argmax != y)[0]]
					false_neg = [((slice == phase).sum() / values[phase]) for phase in range(self.N_CLASSES) if
								 (phase < len(values)) and (values[phase] > 0)]
					values = torch.bincount(out_argmax)
					slice = out_argmax[torch.where(out_argmax != y)[0]]
					false_pos = [((slice == phase).sum() / values[phase]) for phase in range(self.N_CLASSES) if
								 (phase < len(values)) and values[phase] > 0]
					c_acc_val[epoch][i] = 1 - ((torch.sum(torch.stack(false_pos)) + torch.sum(torch.stack(false_neg))) / (len(false_pos) + len(false_neg)))

				# Storing the models into pytorch files to avoid loss of computation
				if epoch % 500 == 499:
					self.logger.info(f"{datetime.now()} storing best model so far")
					torch.save(best_state_dict, f"results/models/{self.name}_e{epoch}.pt")
				if self.remote_store and epoch %100 == 99:
						self.logger_remote.info(f"{datetime.now()} Epoch {epoch}, loss {round(loss_array[epoch].mean(), 4)}, "
				  		f"training accuracy {round(acc_train[epoch].mean() *100, 2)}% ({round((c_acc_train[epoch].mean()) *100, 2)}% weighted), "
				  		f"validation accuracy {round(acc_val[epoch].mean()*100, 2)}% ({round((c_acc_val[epoch].mean()) *100, 2)}% weighted)"
						)
				if self.alpha == 0 and acc_val[:epoch+1].mean(axis=1).argmax() == epoch:
					self.logger.info("Stored state dictionary")
					best_state_dict = copy.deepcopy(self.model.state_dict())
				elif self.alpha > 0 and c_acc_val[:epoch+1].mean(axis=1).argmax() == epoch:
					self.logger.info("Stored state dictionary")
					best_state_dict = copy.deepcopy(self.model.state_dict())

			self.logger.info(f"Epoch {epoch}, loss {round(loss_array[epoch].mean(), 4)}, "
				  f"training accuracy {round(acc_train[epoch].mean() *100, 2)}% ({round((c_acc_train[epoch].mean()) * 100, 2)}% weighted), "
				  f"validation accuracy {round(acc_val[epoch].mean()*100, 2)}% ({round((c_acc_val[epoch].mean()) * 100, 2)}% weighted)"
				  )

		# storing training details to csv
		df_dict = self.config_dict.copy()
		df_dict["weights"] = str(df_dict["weights"])
		results_df = pd.DataFrame({key: np.full(epoch, val) for key, val in df_dict.items()})

		results_df["loss"] = loss_array.mean(axis=1)[:epoch]
		results_df["train_acc"] = acc_train.mean(axis=1)[:epoch]
		results_df["test_acc"] = acc_val.mean(axis=1)[:epoch]
		results_df["weighted_train_acc"] = c_acc_train.mean(axis=1)[:epoch]
		results_df["weighted_test_acc"] = c_acc_val.mean(axis=1)[:epoch]
		results_df["epoch"] = results_df.index
		results_df.to_csv(f"results/csvs/{self.name}.csv", index=False)
		self.logger.info(f"Best achieved test accuracy: {acc_val.mean(axis=1).max()} (epoch {acc_val.mean(axis=1).argmax()})")
		torch.save(best_state_dict, f"results/models/{self.name}.pt")

		if self.remote_store:
			self.logger_remote.info(
				f"Best achieved test accuracy: {acc_val.mean(axis=1).max()} (epoch {acc_val.mean(axis=1).argmax()})")
			results_df.to_csv(f"{self.remote_store}/csvs/{self.name}.csv", index=False)
			torch.save(best_state_dict, f"{self.remote_store}/models/{self.name}.pt")

		return acc_val

if __name__ == '__main__':

	while "src" in os.getcwd():
		os.chdir("..")
		print(f"Changed directory to {os.getcwd()}")

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

	if not os.path.exists('results/logs'):
		os.makedirs('results/logs')

	if not os.path.exists("data/datasets"):
		os.makedirs('data/datasets')

	if not os.path.exists("results/csvs"):
		os.makedirs('results/csvs')

	if not os.path.exists('results/models'):
		os.makedirs('results/models')

	dataset_name = get_name(params['n_flights'], params["n_clusters"], params["n_features"], params["weights"], params["kmeans_iters"])
	if not os.path.exists(f"data/datasets/train_{dataset_name}.pt"):
		print(f"Creating dataset {dataset_name}")
		data_creator = Training_data(n_flights=params['n_flights'],
									 n_clusters=params["n_clusters"],
									 weight_vector=params["weights"],
									 flight_folder=f"data/preprocessed/trajectories_train",
									 ground_truth_folder=f"data/preprocessed/labels_train",
									 n_features=params["n_features"])
		data_creator.store(f"data/datasets", val_ratio=65)

	net = POF_net(params, dev="cpu")
	acc_val = net.train(n_epochs=params["n_epochs"],
						 training=f'data/datasets/train_{dataset_name}.pt',
						 validation=f'data/datasets/val_{dataset_name}.pt')
	model = net.get_model()
	model.load_state_dict(torch.load(f"results/models/{net.get_name()}.pt", map_location=torch.device('cpu')))

