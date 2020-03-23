# import appnope
import json
import numpy as np
import setproctitle
import os
import sys

# from akbinod.Utils.Plotting import Plot

from learners import BaseLearner, RandomForest, LearnerParams, SVM
from learners import DecisionTree, KNearestNeighbors, BoostedTree, NeuralNetwork
from learners import KMeansClusterer, GMMClusterer
from learners.constants import LearnerMode

from Solvers import Queens, FourPeaks, Knapsack, SolverParams, IrisANN

SEED = 0

def test_data():
	p = get_iris_files()
	params = LearnerParams(LearnerMode.pca, learning_target='Species', data_path=p[0])
	lrnr = BaseLearner(params)

def classify_a1(get_coffee):
	ds = "Coffee"
	y = "target"

	if get_coffee:
		files = get_coffee_files(use_reduced=False)
	else:
		files = get_iris_files(use_reduced=False)
		ds = "Iris"
		y = "Species"

	process_files(files, y, 0, LearnerMode.classification, ds_name=ds)

def process_files(paths, target, lrnr_num, mode=LearnerMode.classification, plot_clusters = False, scatter = [], range_k=[], xlabel="", ylabel="", ds_name="", cluster_labels = None):
	ret = None
	results = {}
	plots = []
	for data_path in paths:
		results[data_path] = {}
		params = LearnerParams(mode, learning_target=target, data_path=data_path)
		params.cluster_labels = cluster_labels
		params.split = False

		if mode == LearnerMode.clustering or mode == LearnerMode.silhouette:
			if lrnr_num == 1:
				lrnr = KMeansClusterer(params)
			else:
				lrnr = GMMClusterer(params)

			if mode == LearnerMode.silhouette:
				lrnr.silhouette(range_k=range_k, ds_name=ds_name)
			else:
				ret = lrnr.cluster(range_k, plot_clusters, scatter_axes = scatter,xlabel=xlabel, ylabel=ylabel, ds_name=ds_name)
		elif mode == LearnerMode.pca_num_comp:
			lrnr = BaseLearner(params)
			lrnr.pca_component_analysis(ds_name)
		elif mode == LearnerMode.pca:
			lrnr = BaseLearner(params)
			lrnr.pca_analysis(range_k, ds_name)
		elif mode == LearnerMode.ica_num_comp:
			lrnr = BaseLearner(params)
			lrnr.ica_component_analysis(range_k, ds_name)
		else:
			params.split = True
			lrnr = NeuralNetwork(params)
			lrnr.train()
			res = lrnr.infer()
			results[data_path]['train'] = res[0]
			results[data_path]['test'] = res[1]
			results[data_path]['best_params'] = lrnr.model.best_params_
			plots.append(lrnr.plot_learning_curve())

	print(json.dumps(results))
	for plt in plots:
		plt.show()
	return ret

def get_coffee_files(use_reduced=False):

	paths = ['./data/flavor-coffee.csv']
	if use_reduced:
		paths = ['./data/flavor-coffee-reduced.csv']
	return paths


def get_iris_files(use_reduced=False):
	paths = ['./data/iris-1.csv']
	if use_reduced:
		paths = ['./data/iris-1-reduced.csv']
	return paths

def get_wine_files():
	return ['./data/winequality.csv']


def silhouette_a3(get_coffee, lrnr_num, use_reduced=False):
	''' Silhouette Analysis only '''
	# files = get_wine_files()
	ds = "Coffee"
	if get_coffee:
		files = get_coffee_files(use_reduced=use_reduced)
		y = "target"
		range_k = [50,55,60]
	else:
		files = get_iris_files(use_reduced=use_reduced)
		y = "Species"
		ds = "Iris"
		range_k = [2,3,4,5]

	process_files(files, y, lrnr_num, LearnerMode.silhouette, range_k=range_k, ds_name = ds)

def cluster_a3(get_coffee, lrnr_num, use_reduced = False, plot_clusters = True):
	xlabel = ""
	ylabel = ""

	# determined via silhouette analysis
	num_clusters = 2
	scatter = [2,3]
	ds = "Coffee"
	y = "target"

	if get_coffee:
		files = get_coffee_files(use_reduced=use_reduced)

		plot_clusters = False
		if use_reduced:
			# determined via silhouette analysis
			num_clusters = 55
	else:
		ds = "Iris"
		y = "Species"
		files = get_iris_files(use_reduced=use_reduced)

		if use_reduced:
		# determined via silhouette analysis
			num_clusters = 3
			scatter = [0,1]
			xlabel = "Sepal Length"
			ylabel = "Sepal Width"
		else:
			num_clusters = 2
			scatter = [2,3]
			xlabel = "Petal Length"
			ylabel = "Petal Width"


	return process_files(files, y, lrnr_num, LearnerMode.clustering, plot_clusters=plot_clusters, scatter = scatter,range_k=num_clusters,xlabel=xlabel, ylabel=ylabel, ds_name=ds)

def ica_a3(get_coffee, learner_mode):
	xlabel = ""
	ylabel = ""
	plot_clusters = True
	n_components = 2
	scatter = [2,3]
	ds = "Coffee"
	if get_coffee:
		files = get_coffee_files()
		y = "target"
		plot_clusters = False
		n_components = 100
	else:
		files = get_iris_files()
		y = "Species"
		ds = "Iris"
		xlabel = "Petal Length"
		xlabel = "Petal Width"
		n_components = 2

	process_files(files, y, 0, learner_mode, plot_clusters=plot_clusters, scatter = scatter,range_k=n_components,xlabel=xlabel, ylabel=ylabel, ds_name=ds)

def pca_a3(get_coffee, learner_mode):
	xlabel = ""
	ylabel = ""
	plot_clusters = True
	n_components = 2
	scatter = [2,3]
	ds = "Coffee"
	if get_coffee:
		files = get_coffee_files()
		y = "target"
		plot_clusters = False
		n_components = 100
	else:
		files = get_iris_files()
		y = "Species"
		ds = "Iris"
		xlabel = "Petal Length"
		xlabel = "Petal Width"
		n_components = 2

	process_files(files, y, 0, learner_mode, plot_clusters=plot_clusters, scatter = scatter,range_k=n_components,xlabel=xlabel, ylabel=ylabel, ds_name=ds)

def classify_pca(get_coffee, lrnr_num, with_cluster):
	ds = "Coffee"
	y = "target"
	cluster_labels = None

	# get the reduced file
	if get_coffee:
		files = get_coffee_files(use_reduced=True)
	else:
		ds = "Iris"
		y = "Species"

		files = get_iris_files(use_reduced=True)

	if with_cluster:
		# run the clustering
		cluster_labels = cluster_a3(get_coffee,lrnr_num=lrnr_num,use_reduced=True, plot_clusters=False)

	# run the nn on it
	process_files(files, y, 0, LearnerMode.classification, plot_clusters=False, ds_name=ds, cluster_labels=cluster_labels)
	# print results for comparing to A1
def main(run_name = ""):
	if run_name == "":
		# just use the folder name - that's probably going to be the project/run
		run_name = os.path.dirname(os.path.curdir)

	# this does not show up in mac activity monitor
	# not in top either - that just showed Python
	setproctitle.setproctitle (run_name)
	print(setproctitle.getproctitle())

	# with appnope.nope_scope():
	# cofee analysis
	# kmeans sil
	# silhouette_a3(True, 1)
	# em sil
	# silhouette_a3(True, 2)
	# cluster using KMeans
	# cluster_a3(True,1)
	# Cluster using EM
	# cluster_a3(True, 2)
	# Determine the optimal number of components
	# pca_a3(True, LearnerMode.pca_num_comp)
	# get the new ds after dimensionality reduction
	# pca_a3(True, LearnerMode.pca)
	# Reduced set
	# sil analysis on reduced set : KMeans and EM
	# silhouette_a3(True, 1, True)
	# silhouette_a3(True, 2, True)
	# cluster reduced ds using KMeans
	# cluster_a3(True,1, True)
	# Cluster reduced ds using EM
	# cluster_a3(True, 2, True)
	# recreate the old A1 classification
	# classify_a1(True)
	# classify the reduced dataset
	# classify_pca(True,1,False)
	# classify the reduced dataset after adding cluster labels as a dimension
	# classify_pca(True,1,True)

	# analysis on ICA
	# ica_a3(True,LearnerMode.ica_num_comp)

	# iris analysis
	# silhouette_a3(False, 1)
	# silhouette_a3(False, 2)
	# cluster_a3(False, 1)
	# cluster_a3(False, 2)
	# pca_a3(False, LearnerMode.pca_num_comp)
	# pca_a3(False, LearnerMode.pca)

	# silhouette_a3(False, 1, True)
	# silhouette_a3(False, 2, True)
	# cluster_a3(False,1, True)
	# cluster_a3(False, 2, True)

	# classify_a1(False)
	# classify_pca(False,1,False)
	classify_pca(False,1,True)

if __name__ == "__main__":
	# change this to something that shows the grid search your in
	main("devel project")