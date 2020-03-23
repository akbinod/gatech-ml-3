from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

import matplotlib.cm as cm
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

import akbinod
from akbinod.Utils.TimedFunction import TimedFunction as tf
from learners.constants import LearnerMode


class KMeansClusterer(BaseLearner):
	'''K-Means Clusterer.'''
	def __init__(self, params):
		params.learner_name = "K-Means Clusterer"
		super().__init__(params)

	def silhouette(self, range_k = [], ds_name=""):
		if len(range_k) == 0:
			# range_n_clusters = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
			# range_n_clusters = [42, 43, 44, 45, 46, 48, 50, 52, 54, 56, 58, 60]
			# range_n_clusters = [100,200,250,275,300]
			range_n_clusters = [2, 3, 4, 5, 6, 8]
		else:
			range_n_clusters = range_k

		for n_clusters in range_n_clusters:

			# Initialize the clusterer with n_clusters value and a random generator
			# seed of 10 for reproducibility.
			clusterer = KMeans(n_clusters=n_clusters, random_state=10)
			# clusterer =
			cluster_labels = clusterer.fit_predict(self.X)
			self.sihouette_analysis(n_clusters, cluster_labels, ds_name)

	@tf(True)
	def cluster(self, num_clusters, plot_clusters = False, scatter_axes = [], range_k = [],xlabel="", ylabel="", ds_name=""):

		# Initialize the clusterer with n_clusters value and a random generator
		# seed of 10 for reproducibility.
		clusterer = KMeans(n_clusters=num_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(self.X)
		self.cluster_scores(cluster_labels, ds_name)

		if plot_clusters:
			# 2nd Plot showing the actual clusters formed
			colors = cm.nipy_spectral(cluster_labels.astype(float) / num_clusters)
			# ax2.
			plt.scatter(self.X[:, scatter_axes[0]], self.X[:, scatter_axes[1]]
						, marker='.', s=30, lw=0, alpha=0.7
						, c=colors, edgecolor='k')

			# Labeling the clusters
			centers = clusterer.cluster_centers_
			# Draw white circles at cluster centers
			plt.scatter(centers[:, 0], centers[:, 1], marker='o',
						c="white", alpha=1, s=200, edgecolor='k')

			for i, c in enumerate(centers):
				plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
							s=50, edgecolor='k')

			plt.title("Visualization of clustered data.")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)

			plt.suptitle(f"{self.learner_params.learner_name} : {ds_name}",	fontsize=14, fontweight='bold')

			plt.show()

		return cluster_labels