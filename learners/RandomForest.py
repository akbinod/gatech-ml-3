from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from learners.constants import LearnerMode

class RandomForest(BaseLearner):
	'''Random Forest class.'''
	def __init__(self, params):
		params.learner_name = "random_forest"
		super().__init__(params)

		self.clf = RandomForestClassifier()
		# 6. Declare hyperparameters to tune
		self.hyperparameters = {
					'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2']
					,'randomforestclassifier__max_depth': [None, 5, 4, 3]
					}

		pipeline = make_pipeline(self.clf)
		self.model = GridSearchCV(pipeline, self.hyperparameters, scoring = 'f1_weighted', cv=self.learner_params.cv)






