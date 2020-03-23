from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(BaseLearner):
	'''Neural Network.'''
	def __init__(self, params):
		params.learner_name = "Neural Network"
		super().__init__(params)
		self.clf = MLPClassifier(max_iter=200000)
		# print(self.clf.get_params().keys())
		self.build_pipeline()

	def build_pipeline(self):
		# self.hyperparameters = {
		# 	'mlpclassifier__solver' : ['sgd', 'adam']
		# 	, 'mlpclassifier__learning_rate' : ['adaptive', 'invscaling']
		# 	, 'mlpclassifier__hidden_layer_sizes' : [(6, 24, 4), (6, 24), (6)]
		# 	, 'mlpclassifier__max_iter' : [200_000]
		# 	}
		self.hyperparameters = {
			'mlpclassifier__solver' : ['sgd']
			, 'mlpclassifier__learning_rate' : ['adaptive']
			, 'mlpclassifier__hidden_layer_sizes' : [(6, 24, 4), (6, 24), (6)]
			, 'mlpclassifier__max_iter' : [200_000]
			}

		pipeline = make_pipeline(self.clf)
		self.model = GridSearchCV(pipeline, self.hyperparameters, scoring = 'f1_weighted', cv=self.learner_params.cv)
