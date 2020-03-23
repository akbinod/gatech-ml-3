import mlrose_hiive as mlrose
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import f1_score

# from Solvers.BaseSolver import BaseSolver
# from Solvers.RandomHIllClimbing import RandomHillClimbing
# from Solvers.SimulatedAnnealing import SimulatedAnnealing
# from Solvers.GeneticAlgorithm import GeneticAlgorithm
# from Solvers.Mimic import Mimic
# from Solvers.SolverParams import SolverParams

from akbinod.Utils.TimedFunction import TimedFunction
import copy
from time import process_time

class IrisANN():
	def __init__(self, params):
		# super().__init__(params)

		self.name = "IrisANN"
		self.params = params
		self.load_data()

	def dump_models(self, models):
		print(f"algorithm\tfrom\ttrain_f1\ttest_f1\ttime")
		for pa in models:

			# Initialize neural network object and fit object
			t1 = process_time()
			pa.model.fit(self.X_train_scaled, self.y_train_hot)
			t2 = process_time()
			pa.time = t2 - t1
			# Predict labels for train set and assess accuracy
			y_train_pred = pa.model.predict(self.X_train_scaled)
			pa.y_train_f1 = f1_score(self.y_train_hot, y_train_pred,average='weighted')
			# Predict labels for test set and assess accuracy
			y_test_pred = pa.model.predict(self.X_test_scaled)
			pa.y_test_f1 = f1_score(self.y_test_hot, y_test_pred,average='weighted')

			print(f"{pa.algorithm}\t{pa.name}\t{round(pa.y_train_f1,4)}\t{round(pa.y_test_f1,4)}\t{round(pa.time,4)}")

	TimedFunction(True)
	def benchmark(self):

		models = self.get_models_sa()
		self.dump_models(models)

		# UNCOMMENT THE FOLLOWING LINES FOR A FULL RUN
		# models = self.get_models_rhc()
		# self.dump_models(models)

		# models = self.get_models_ga()
		# self.dump_models(models)

	def load_data(self):
		# Split data into training and test sets
		# Load the Iris dataset
		data = load_iris()
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
										data.data
										, data.target
										,test_size = 0.2, random_state = 3)

		# Normalize feature data
		scaler = MinMaxScaler()

		self.X_train_scaled = scaler.fit_transform(self.X_train)
		self.X_test_scaled = scaler.transform(self.X_test)

		# One hot encode target values
		one_hot = OneHotEncoder()

		self.y_train_hot = one_hot.fit_transform(self.y_train.reshape(-1, 1)).todense()
		self.y_test_hot = one_hot.transform(self.y_test.reshape(-1, 1)).todense()

	def get_models_sa(self):

		pas = []
		# start with the HP tuned for SA in part 1
		pa = copy.deepcopy(self.params)
		pa.name = "Queens - SA"
		pa.algorithm = "simulated_annealing"
		# pa.decay_schedule = mlrose.GeomDecay()
		pa.random_state = None
		pa.max_iters = 2500
		pa.max_attempts = 250
		pas.append(pa)

		pa = copy.deepcopy(self.params)
		pa.name = "Knapsack - SA"
		pa.algorithm = "simulated_annealing"
		# pa.decay_schedule = mlrose.GeomDecay()
		pa.random_state = None
		pa.max_iters = 20_000
		pa.max_attempts = 2500
		pas.append(pa)

		pa = copy.deepcopy(self.params)
		pa.name = "Peaks - SA"
		pa.algorithm = "simulated_annealing"
		# pa.decay_schedule = mlrose.GeomDecay()
		pa.random_state = None
		pa.max_iters = 2500
		pa.max_attempts = 250
		pas.append(pa)

		for pa in pas:
			pa.model = mlrose.NeuralNetwork(
								hidden_nodes = [6,24]
								, activation = 'relu'
								, algorithm = pa.algorithm
								, max_iters = pa.max_iters
								, max_attempts = pa.max_attempts
								, random_state = pa.random_state
								, bias = True
								, is_classifier = True	#based on problem
								, learning_rate = 0.1	#based on Ass1
								, early_stopping = True
								, clip_max = 5
								)

		return pas

	def get_models_rhc(self):

		pas = []
		# start with the HP tuned for SA in part 1
		pa = copy.deepcopy(self.params)
		pa.name = "Queens - RHC"
		pa.algorithm = "random_hill_climb"
		pa.random_state = None
		pa.restarts = 250
		pa.max_iters = 250
		pa.max_attempts = 250
		pas.append(pa)

		pa = copy.deepcopy(self.params)
		pa.name = "Knapsack - RHC"
		pa.algorithm = "random_hill_climb"
		pa.random_state = None
		pa.restarts = 250
		pa.max_iters = 20_000
		pa.max_attempts = 2500
		pas.append(pa)

		pa = copy.deepcopy(self.params)
		pa.name = "Peaks - RHC"
		pa.algorithm = "random_hill_climb"
		pa.random_state = None
		pa.restarts = 250
		pa.max_iters = 250
		pa.max_attempts = 250
		pas.append(pa)

		for pa in pas:
			pa.model = mlrose.NeuralNetwork(
								hidden_nodes = [6,24]
								, activation = 'relu'
								, algorithm = pa.algorithm
								, max_iters = pa.max_iters
								, max_attempts = pa.max_attempts
								, random_state = pa.random_state
								, restarts=pa.restarts
								, bias = True
								, is_classifier = True	#based on problem
								, learning_rate = 0.1	#based on Ass1
								, early_stopping = True
								, clip_max = 5
								)
		return pas

	def get_models_ga(self):

		pas = []
		# start with the HP tuned for SA in part 1
		pa = copy.deepcopy(self.params)
		pa.name = "Queens - GA"
		pa.algorithm = "genetic_alg"
		pa.random_state = None
		pa.pop_size = 1000
		pa.pop_breed_percent = 0.50
		pa.mutation_prob = 0.25
		pa.max_iters = 20_000
		pa.max_attempts = 250
		pas.append(pa)

		pa = copy.deepcopy(self.params)
		pa.name = "Knapsack - GA"
		pa.algorithm = "genetic_alg"
		pa.random_state = None
		pa.pop_size = 1000
		pa.pop_breed_percent = 0.25
		pa.mutation_prob = 0.25
		pa.max_iters = 20_000
		pa.max_attempts = 250
		pas.append(pa)

		pa = copy.deepcopy(self.params)
		pa.name = "Peaks - GA"
		pa.algorithm = "genetic_alg"
		pa.random_state = None
		pa.pop_size = 1000
		pa.pop_breed_percent = 0.50
		pa.mutation_prob = 0.25
		pa.max_iters = 20_000
		pa.max_attempts = 250

		pas.append(pa)

		for pa in pas:
			pa.model = mlrose.NeuralNetwork(
								hidden_nodes = [6,24]
								, activation = 'relu'
								, algorithm = pa.algorithm
								, max_iters = pa.max_iters
								, max_attempts = pa.max_attempts
								, random_state = pa.random_state
								, pop_size = pa.pop_size
								# , pop_breed_percent = pa.pop_breed_percent
								, mutation_prob= pa.mutation_prob
								, bias = True
								, is_classifier = True	#based on problem
								, learning_rate = 0.1	#based on Ass1
								, early_stopping = True
								, clip_max = 5
								)
		return pas

	def __str__(self):
		return "IrisANN"