from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class BoostedTree(BaseLearner):
	'''Boosting class.'''
	def __init__(self, params):
		params.learner_name = "BoostedTree"
		super().__init__(params)
		self.clf = AdaBoostClassifier(
    				DecisionTreeClassifier(max_depth=100)
    			,algorithm="SAMME"
				)
		# print(self.clf.get_params().keys())
		self.build_pipeline()

	def build_pipeline(self):
		# self.build_pipeline_restricted()
		self.hyperparameters = {
					'adaboostclassifier__n_estimators' : [50, 25, 5]
					, 'adaboostclassifier__learning_rate' : [1, 1.5, 2]
					# ,'base_estimator__max_depth': ['Auto',16, 8, 4]
					}
		pipeline = make_pipeline(self.clf)
		self.model = GridSearchCV(pipeline, self.hyperparameters, scoring = 'f1_weighted', cv=self.learner_params.cv)
