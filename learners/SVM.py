from learners import BaseLearner, constants
import numpy as np
np.random.seed(constants.SEED)

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class SVM(BaseLearner):
	'''Support Vector Machine.'''
	def __init__(self, params):
		params.learner_name = "SVM"
		super().__init__(params)
		self.clf = SVC()
		print(self.clf.get_params().keys())
		self.build_pipeline()

	def build_pipeline(self):
		# self.build_pipeline_restricted()
		self.hyperparameters = {
					'svc__kernel' : ['linear', 'poly', 'rbf']
					, 'svc__degree' : [3, 5, 9]
					}
		pipeline = make_pipeline(self.clf)
		self.model = GridSearchCV(pipeline, self.hyperparameters, scoring = 'f1_weighted', cv=self.learner_params.cv)

	@staticmethod
	def supports_auto():
		return False
