# SPDX-FileCopyrightText: 2023 German Aerospace Center
#
# SPDX-License-Identifier: MIT

'''
Author: Emy Arts (emy.arts@dlr.de)

Training the flight phase estimating network (network.py) with different hyperparameter configurations (command_configs).
Results of each epoch are printed and the final training statistics csv file stored in results/csv for each network.
'''


from pof_functions.create_dataset import Training_data, get_name
from sklearn.model_selection import ParameterGrid
from pof_functions.network import POF_net
import argparse
import os
import gc

val_split = 65

command_configs = {
	"n_clusters": [160],
	"n_features": [6, 7],
	"batch_size": [16],
	"weights": [[0.7, 0.8, 0.1]],
	"n_layers": [2],
	"learning_rate": [0.01],
	"hidden_dim": [16],
	"n_epochs": [3500],
	"alpha": [0, 3],
	"kmeans_iters": [0, 100]
}

if __name__ == '__main__':

	while "src" in os.getcwd():
		os.chdir("..")
		print(f"Changed directory to {os.getcwd()}")

	parser = argparse.ArgumentParser(description="Run a parameter gridsearch of the flight phase estimating network")
	parser.add_argument("--id", type=int, help="id of the specific run")
	parser.add_argument("--tot_exps", type=int, default=3, help="The total amount of parallel experiments")
	parser.add_argument("--trials", type=int, default=3, help="The amount of times the same network is trained.")
	parser.add_argument("--device", default="cuda", help="The device to train on cuda or cpu")
	parser.add_argument("--n_epochs", default=command_configs["n_epochs"], help="Number of training epochs")

	args = parser.parse_args()
	exp_id = args.id
	tot_exps = args.tot_exps
	n_trials = args.trials
	device = args.device

	if not os.path.exists('results/logs'):
		os.makedirs('results/logs')

	if not os.path.exists("data/datasets"):
		os.makedirs('data/datasets')

	if not os.path.exists("results/csvs"):
		os.makedirs('results/csvs')

	if not os.path.exists('results/models'):
		os.makedirs('results/models')

	grid = ParameterGrid(command_configs)
	n_nets = len(grid)
	for trial in range(0, n_trials):
		for idx, params in enumerate(grid):
			params['trial'] = trial
			idx += trial * n_nets
			if idx % tot_exps == exp_id and not(params['n_features'] == 6 and params['kmeans_iters'] == 0):
				dataset_name = get_name(params["n_clusters"], params["n_features"], params["weights"], params["kmeans_iters"])
				if not os.path.exists(f"data/datasets/train_{dataset_name}.pt"):
					data_creator = Training_data(n_clusters=params['n_clusters'],
												 weight_vector=params['weights'],
												 n_features=params['n_features'],
												 flight_folder=f"data/preprocessed/trajectories_train",
												 ground_truth_folder=f"data/preprocessed/labels_train",
												 kmeans_iters=params["kmeans_iters"]
												 )
					data_creator.store(folder="data/datasets", val_ratio=val_split)
				model = POF_net(config_dict=params, dev=device)
				try:
					if not os.path.exists(f"results/csvs/{model.get_name()}.csv"):
						model.train(n_epochs=params["n_epochs"], training=f'data/datasets/train_{dataset_name}.pt', validation=f'data/datasets/val_{dataset_name}.pt')
					gc.collect()
				except:
					print(f"Could not train network {model.get_name()}!")
					pass