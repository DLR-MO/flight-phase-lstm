'''
Author Emy Arts (emy.arts@dlr.de)

K-means segmentation segments flights based on the amount of change in their trajectory variables.
Main analyses different weights for the segmentation.
'''

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import re
import itertools
import math
import logging

VAL_LIMITS = {
    # Altitude and speed taken from A320 and B737 docs
    # 'ts':[4, 1800], # artificially set segments to max 30 min
    'alt': [0, 41000], # ft Both A320 and B737 have same ceiling
    'spd': [0, 470], # spd (kts) Both A320 and B737 have same MMO (max mach operation)
    'roc': [-10000, 10000], # roc (fpm) ICAO docs (https://www.icao.int/Meetings/anconf12/Document%20Archive/9863_cons_en.pdf)
}

def norm(array:np.ndarray, column:str) -> np.ndarray:
	return (array - VAL_LIMITS[column][0]) / (VAL_LIMITS[column][1] - VAL_LIMITS[column][0])


def alpha_numeric_sort(l:list) -> list:
	""" Sort the given iterable in the way that humans expect."""
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key = alphanum_key)

number_to_label = {
    0: "NA",
    1: "taxi",
    2: "take off",
    3: "initial climb",
    4: "climb",
    5: "cruise",
    6: "descent",
    7: "approach",
    8: "landing"
}

colormap = {
        "NA": "red",
        "taxi": "black",
        "take off": "pink",
        "initial climb": "yellow",
        "climb": "green",
        "cruise": "blue",
        "descent": "orange",
        "approach": "brown",
        "landing": "cyan",
    }



def kmeans_segmentation(x:np.ndarray, n_mu:int, weights:list, max_iters = 100)->(np.ndarray, np.ndarray):
	'''
	Segment x into n_mu segments of variable length using k-means clustering and enforcing continuity

	:param x: multi-dimensional signal to be segmented
	:param n_mu: number of segments
	:param weights: weights in [0, 1] indicating influence of the dimensions
	:return: an array with the segment number for each data point, mean of each segment
	'''

	# Uniform initialisation
	dims = len(x[0])
	n = len(x)
	c = np.zeros(n, dtype=np.int)
	same_class = math.floor(n/n_mu)
	rest = n % n_mu
	slice_start = 0
	for i in range(n_mu):
		if i < rest:
			slice_end = slice_start + same_class + 1
		else:
			slice_end = slice_start + same_class
		c[slice_start:slice_end] = i
		slice_start = slice_end

	def dist(x1:np.ndarray, x2:np.ndarray, w = weights)->float:
		'''
		Compute the Eucledian distance between x1 and x2 using the w weights

		:param x1: a datapoint
		:param x2: a second datapoint with same dimension as x1
		:param w: weights for each dimension
		:return: Eucledian distance between x1 and x2
		'''

		assert len(x1) == len(x2)
		assert len(x1) == len(w)
		dist = 0
		for i in range(len(x1)):
			dist += weights[i] * np.power((x1[i] - x2[i]), 2)
		dist = np.sqrt(dist)
		return dist

	def get_means(c: np.ndarray, x_inner: np.ndarray = x, dim: int = dims, n_clusters: int = n_mu)->np.ndarray:
		'''
		Compute the segment means with the current segmentation

		:param c: array with segment index for each datapoint
		:param x_inner: multi-dimension signal
		:param dim: number of dimensions of signal
		:param n_clusters: number of segments
		:return: array with mean values of each dimension for each segment
		'''

		means = np.zeros((n_clusters, dim))
		counts = np.zeros(n_clusters)
		for i, x_i in enumerate(x_inner):
			means[c[i]] += x_i
			counts[c[i]] += 1
		counts = np.where(counts == 0, 1, counts)
		for i, m in enumerate(means):
			m /= counts[i]
		return means

	mu = get_means(c)
	# Core
	for iteration in range(max_iters):
		new_c = c.copy()
		counts = pd.value_counts(new_c)
		for idx in range(1, n-1):
			if counts[c[idx+1]] < 1800 and counts[c[idx-1]] <1800: # Segments should not be longer than half an hour
				if c[idx+1] != c[idx] and counts[c[idx]] > 4:
					dist_next = dist(x[idx], mu[c[idx+1]])
					dist_curr = dist(x[idx], mu[c[idx]])
					if dist_next < dist_curr:
						new_c[idx] = c[idx + 1]
				elif c[idx-1] != c[idx] and counts[c[idx]] > 4:
					dist_curr = dist(x[idx], mu[c[idx]])
					dist_prev = dist(x[idx], mu[c[idx-1]])
					if dist_prev < dist_curr:
						if new_c[idx-1] == c[idx-1]:
							new_c[idx] = c[idx-1]
						# In case two consecutive elements want to switch only the one with biggest difference does
						elif dist(x[idx-1], mu[c[idx-1]]) - dist(x[idx-1], mu[c[idx]]) < dist_curr - dist_prev:
							new_c[idx] = c[idx-1]
							new_c[idx-1] = c[idx-1]
		if any(new_c != c):
			c = new_c.copy()
			mu = get_means(c)
		else:
			# print("Stopped at iteration " + str(iteration))
			return mu, c
	print("Reached max iterations")
	return mu, c

def segmentation_weight_evaluation(flights:list, labels:list, weights:list, plot=False, n_cluster=90):
	'''
	Function that computes the best possible accuracy for the flights when using weights

	:param flights: pandas dataframes with trajectories
	:param labels: pandas dataframes with ground truth labels
	:param weights: weights ([0, 1]) to apply for segmentation
	:param plot: show the identified segments
	:param n_cluster: number of segments
	:return:
	'''

	count_stats = 0
	error = []
	mean_err = 0
	max_err = 0
	corrects = 0
	phase_lens = np.zeros(8)
	for flight_n in range(len(flights)):
		label = labels[flight_n]
		flight = flights[flight_n]
		x = [norm(flight[col].copy(), col) for col in flight.columns if col in VAL_LIMITS.keys()]
		x = np.array(x).transpose()
		phase_counts = np.bincount(label["phase"])

		ts = flight['ts']
		alts = flight['alt']

		## K-means
		_, twindows = kmeans_segmentation(x, n_mu=n_cluster, weights=weights)
		flight['cluster'] = twindows
		counts = pd.value_counts(twindows)
		count_stats = np.add(count_stats, [counts.max(), counts.min(), counts.mean()])
		flight["phase"] = 0
		for i, t in enumerate(twindows):
			if t != twindows[i - 1]:
				if counts[t] > 0:
					l_counts = label['phase'][i:(i + counts[t])].value_counts()
					if not l_counts.empty:
						flight.iloc[i:(i + counts[t]), flight.columns.get_loc('phase')] = l_counts.idxmax()
		phase_lens += np.array([(abs(flight["phase"][label["phase"]==i] - label["phase"][label["phase"]==i]) > 0).sum() / phase_counts[i] for i in range(8)])
		mean_err = (abs(flight["phase"] - label["phase"]) > 0).sum() / len(label)

		correct_idx = []
		missed_idx = []
		if plot:
			fig, ax = plt.subplots(2, figsize=(20, 10))
			ax[0].scatter(ts, alts, s=1, c=np.mod(twindows, 2), cmap='viridis')
			ax[1].scatter(ts, alts, s=1, c=label['phase'], cmap='viridis')
			for idx in correct_idx:
				ax[1].axvline(ts[idx], color='green', linestyle='--')
			for idx in missed_idx:
				ax[1].axvline(ts[idx], color='red', linestyle='--')
			plt.show()
	count_stats /= len(flights)
	mean_err /= len(flights)
	max_err /= len(flights)
	corrects /= len(flights)
	phase_lens /= len(flights)
	print(f"{weights}, window lengths: {np.round(count_stats, 2)}, average relative error {round(mean_err*100, 4)}%, average maximum relative error {round(max_err*100, 5)}%")
	return mean_err, phase_lens

def label_clusters(flights:list, labels:list, weights:list, plot=True, n_cluster=90)->(np.ndarray, float):
	'''
	Segments flights with k-means and label the segments with the most occuring ground truth label in that segment

	:param flights: pandas dataframes with trajectories
	:param labels: pandas dataframes labels
	:param weights: weights (in range [0,1]) applied for segmentation
	:param plot: show segmentation (True) or not (False)
	:param n_cluster: number of clusters
	:return: segment labels of all flights, error introduced by segmentation
	'''
	clustered_labels = []
	diffs = np.zeros(len(flights))
	k_means_iters = np.zeros(len(flights))
	for flight_n in range(len(flights)):
		label = labels[flight_n]
		flight = flights[flight_n]
		x = [norm(flight[col].copy(), col) for col in flight.columns if col in VAL_LIMITS.keys()]
		x = np.array(x).transpose()
		for i_w, w_i in enumerate(weights):
			x[:, i_w] = w_i * x[:, i_w]
		ts = flight.ts
		alts = flight.alt

		## K-means
		_, twindows, k_means_iters[flight_n] = kmeans_segmentation(x, n_mu=n_cluster)
		flight['cluster'] = twindows
		counts = pd.value_counts(twindows)
		labs = np.zeros(len(twindows))
		for i, t in enumerate(twindows):
			if t != twindows[i-1]:
				l_counts = label['phase'][i:(i+counts[t])].value_counts()
				print(l_counts)
				labs[i:(i+counts[t])] = l_counts.idxmax()

		if plot:
			colors_c = [colormap[number_to_label[l]] for l in labs]
			colors_l = [colormap[number_to_label[l]] for l in label['phase']]

			fig, ax = plt.subplots(2, figsize=(20, 10))
			ax[0].scatter(ts, alts, s=1, c=colors_c, cmap='viridis')
			ax[1].scatter(ts, alts, s=1, c=colors_l, cmap='viridis')
			ax[0].set_title("Clustered")
			ax[1].set_title("Original")
			plt.show()
		clustered_labels.append(labs)
		diffs[flight_n] = sum((label['phase'] != labs))/len(labs)
	return clustered_labels, diffs


if __name__ == '__main__':

	folder = "../../data/preprocessed"

	parser = argparse.ArgumentParser()
	parser.add_argument("--trajectory_folder", default="../../data/preprocessed/trajectories_train", help="The folder where the trajectories are stored.")
	parser.add_argument("--label_folder", default="../../data/preprocessed/labels_train", help="The folder where the labels are stored.")

	args = parser.parse_args()
	t_folder = args.trajectory_folder
	l_folder = args.label_folder

	labels = []
	flights = []
	for f_idx, file in enumerate(alpha_numeric_sort(os.listdir(l_folder))):
		l = pd.read_csv(f"{l_folder}/{file}")
		labels.append(l)
		flights.append(pd.read_csv(f"{t_folder}/{file}"))

	# List of list of weights for each dimension
	weight_vals = [
		[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # alt
		[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # spd
		[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # roc

	]

	weights_list = [ws for ws in list(itertools.product(*weight_vals))]

	acc = np.zeros(len(weights_list))
	cluster_numbers = [160]
	weights_list = [ws for ws in weights_list if not tuple(np.divide(ws, 2)) in weights_list]
	print(weights_list)
	acc = np.zeros((len(cluster_numbers), len(weights_list)))
	worst = np.zeros((len(cluster_numbers), len(weights_list), 8))
	avg_wind = np.zeros((len(cluster_numbers), len(weights_list)))
	logger = logging.getLogger("Net logger")
	print("Creating log file")
	fh = logging.FileHandler(f"../kmeans_opt_1307_{cluster_numbers}_uni.log")
	logger.setLevel(logging.INFO)
	logger.addHandler(fh)
	for i, cs in enumerate(cluster_numbers):
		print(cs)
		for j, w in enumerate(weights_list):
			acc[i][j], worst[i][j] = segmentation_weight_evaluation(flights=flights, labels=labels, weights=w, plot=False, n_cluster=cs)
			logger.info(f"{cs}, {w}: accuracy {acc[i][j]}")
	for i in range(len(cluster_numbers)):
		print(f"{cluster_numbers[i]}: best acc ({acc[i].min()}, {worst[i][np.argmin(acc[i])]}) with {weights_list[np.argmin(acc[i])]}")
		print(f"{cluster_numbers[i]}: min worst ({acc[i][np.argmin(worst[i].max(axis=1))]}, {worst[i].max(axis=1).min()}) with {weights_list[np.argmin(worst[i])]}")