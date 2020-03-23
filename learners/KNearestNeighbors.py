from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbors(BaseLearner):
	'''K Nearest Neighbors class.'''
	def __init__(self, params):
		params.learner_name = "KNN"
		super().__init__(params)
		self.clf = KNeighborsClassifier()
		self.build_pipeline()

	def build_pipeline(self):
		# 6. Declare hyperparameters to tune
		self.hyperparameters = {
					'kneighborsclassifier__n_neighbors' : [5,7,9]
					,'kneighborsclassifier__weights': ['uniform', 'distance']
					, 'kneighborsclassifier__metric': ['minkowski', 'manhattan']
					}
		pipeline = make_pipeline(self.clf)
		self.model = GridSearchCV(pipeline, self.hyperparameters, scoring = 'f1_weighted', cv=self.learner_params.cv)

