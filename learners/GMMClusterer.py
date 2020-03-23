
from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

import matplotlib.cm as cm
from matplotlib import pyplot as plt

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
# import seaborn as sns
# sns.set()

import akbinod
from akbinod.Utils.TimedFunction import TimedFunction as tf
from learners.constants import LearnerMode


class GMMClusterer(BaseLearner):
	'''Expectation Maximization Clusterer.'''
	def __init__(self, params):
		params.learner_name = "Expectation Maximization Clusterer"
		super().__init__(params)
	def bci_analysis(self, n_components):
			# use the following 3 lines for coffee, and the next 3 for Iris
			# models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(self.X.toarray()) for n in n_components]
			# plt.plot(n_components, [m.bic(self.X.toarray()) for m in models], label='BIC')
			# plt.plot(n_components, [m.aic(self.X.toarray()) for m in models], label='AIC')
			models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(self.X) for n in n_components]
			plt.plot(n_components, [m.bic(self.X) for m in models], label='BIC')
			plt.plot(n_components, [m.aic(self.X) for m in models], label='AIC')

			plt.legend(loc='best')
			plt.xlabel('n_components')

	def silhouette(self, range_k = [], ds_name=""):
		bic = 0
		X = self.not_sparse_X

		if len(range_k) == 0:
			# trying to figure out how many components to use
			n_components = np.arange(2, 10)
		else:
			n_components = range_k
		if bic:
			n_components = np.arange(1, 21)
			self.bci_analysis(n_components)
		else:
			# silhouette analysis

			for n in n_components:
				gmm = GaussianMixture(n)
				gmm.fit(X)
				labels = gmm.predict(X)
				self.sihouette_analysis(n,labels, ds_name)

	@tf(True)
	def cluster(self, num_clusters, plot_clusters = True, scatter_axes = [], title = "", xlabel="", ylabel="", ds_name=""):
		X = self.not_sparse_X


		gmm = GaussianMixture(num_clusters)
		gmm.fit(X)
		labels = gmm.predict(X)
		self.cluster_scores(labels, ds_name)

		if plot_clusters:

			plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.suptitle(f"{self.learner_params.learner_name} : {ds_name}",	fontsize=14, fontweight='bold')

			plt.show()
		return labels

