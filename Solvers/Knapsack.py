import mlrose_hiive as mlrose
import numpy as np
from Solvers.BaseSolver import BaseSolver
from Solvers.RandomHIllClimbing import RandomHillClimbing
from Solvers.SimulatedAnnealing import SimulatedAnnealing
from Solvers.GeneticAlgorithm import GeneticAlgorithm
from Solvers.Mimic import Mimic

from akbinod.Utils.TimedFunction import TimedFunction
# from mlrose_hiive.generators import KnapsackGenerator


import copy

class Knapsack(BaseSolver):
	def __init__(self, params):
		super().__init__(params)

		self.name = "Knapsack"
		self._fitness_label = "score"

		# Define initial state
		self.init_state = None
		self.maximize = True
		self.init_empirical_hp()

		# Define the problem, and initialize custom fitness function object
		# This delegates to the builtin but gives us the opportunity
		# to capture fitness values as they are generated. For use in comparison.

		self.problem, weights, values = Knapsack.generate(903549491
														,number_of_items_types= self.params.items)
		self.fitness_delegate = mlrose.Knapsack(weights,values) #will be used by custom fitness fn
		self.problem.fitness_fn = mlrose.CustomFitness(self.fitness_fn)

	def init_empirical_hp(self):
		# these truths we hold to be self evident (come from HP tuning)
		# at the 16 Queens level
		# set up whatever we know about SA - good, done
		pa = copy.deepcopy(self.params)
		pa.decay_schedule = mlrose.GeomDecay()
		pa.random_state = None
		pa.max_iters = np.inf
		pa.max_attempts = 2500
		self.algorithms.append(SimulatedAnnealing(pa))

		# set up whatever we know about RHC
		pa = copy.deepcopy(self.params)
		pa.random_state = None
		pa.restarts = 250
		pa.max_iters = np.inf
		pa.max_attempts = 2500
		self.algorithms.append(RandomHillClimbing(pa))

		# set up whatever we know about GA
		pa = copy.deepcopy(self.params)
		pa.pop_size = 1000
		pa.pop_breed_percent = 0.25
		pa.mutation_prob = 0.25
		pa.max_iters = np.inf
		pa.max_attempts = 250
		pa.random_state = None
		self.algorithms.append(GeneticAlgorithm(pa))

		# set up whatever we know about Mimic
		pa = copy.deepcopy(self.params)
		pa.pop_size = 600
		pa.keep_pct = 0.1
		pa.noise = 0.1
		pa.max_iters = np.inf
		pa.max_attempts = 50
		pa.random_state = None
		self.algorithms.append(Mimic(pa))

	@property
	def fitness_label(self):
		return self._fitness_label

	def __str__(self):
		return self.name + "-" + str(self.params.items)

	@staticmethod
	def generate(seed, number_of_items_types=10,
				max_item_count=5, max_weight_per_item=25,
				max_value_per_item=10, max_weight_pct=0.6,
				multiply_by_max_item_count=True):
		np.random.seed(seed)
		weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
		values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)
		problem = mlrose.KnapsackOpt(length=number_of_items_types
									,maximize=True, max_val=max_item_count
									,weights=weights, values=values
									,max_weight_pct=max_weight_pct
									,multiply_by_max_item_count=multiply_by_max_item_count
									)
		return problem, weights, values