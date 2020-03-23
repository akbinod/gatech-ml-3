import numpy as np
from matplotlib import pyplot as plt

class BaseAlgorithm():
	def __init__(self, params):
		self.params = params

		self.fitness_scores = []
		self.iteration_scores = []
		self.best_state = None
		self.best_fitness = None

	def __str__(self):
		ret = str(self.name)
		# if self.best_fitness is not None:
		# 	ret = ret + " - best: " + str(self.best_fitness)

		return ret

	def tune(self, problem, init_state):
		# Implementors must override this function.
		# Must return best_state, best_fitness,
		# iteration_scores[] (same as curve returned by the algorithm),
		# fitness_scores[] (accumulation of fitness scores)
		raise NotImplementedError()

	def solve(self, problem, init_state):
		# Implementors must override this function.
		# Must return best_state, best_fitness,
		# iteration_scores[] (same as curve returned by the algorithm),
		raise NotImplementedError()
