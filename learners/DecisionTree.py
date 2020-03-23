from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import tree

class DecisionTree(BaseLearner):
	'''Decision Tree class.'''
	def __init__(self, params):
		params.learner_name = "decision_tree"
		super().__init__(params)
		self.clf = tree.DecisionTreeClassifier()
		self.build_pipeline()

	def build_pipeline(self):
		# 6. Declare hyperparameters to tune
		self.hyperparameters = {
					'decisiontreeclassifier__max_features' : ['auto', 'sqrt']
					,'decisiontreeclassifier__max_depth': [None, 16, 8, 4]
					}
		pipeline = make_pipeline(self.clf)
		self.model = GridSearchCV(pipeline, self.hyperparameters, scoring = 'f1_weighted', cv=self.learner_params.cv)

