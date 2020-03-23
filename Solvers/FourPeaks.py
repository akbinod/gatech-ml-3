import mlrose_hiive as mlrose
import numpy as np
from Solvers.BaseSolver import BaseSolver
from Solvers.RandomHIllClimbing import RandomHillClimbing
from Solvers.SimulatedAnnealing import SimulatedAnnealing
from Solvers.GeneticAlgorithm import GeneticAlgorithm
from Solvers.Mimic import Mimic

from akbinod.Utils.TimedFunction import TimedFunction

import copy

class FourPeaks(BaseSolver):
	def __init__(self, params):
		super().__init__(params)

		self.name = "FourPeaks"
		self._fitness_label = "score"

		# Define initial state
		self.init_state = None
		self.maximize = True
		self.init_empirical_hp()

		# Define the problem, and initialize custom fitness function object
		# This delegates to the builtin but gives us the opportunity
		# to capture fitness values as they are generated. For use in comparison.
		self.fitness_delegate = mlrose.FourPeaks() #will be used by custom fitness fn
		self.problem = mlrose.DiscreteOpt(length = self.params.length
										, fitness_fn = mlrose.CustomFitness(self.fitness_fn)
										, maximize = self.maximize
										, max_val = 2)

	def init_empirical_hp(self):
		# these truths we hold to be self evident (come from HP tuning)
		# at the 16 Queens level
		# set up whatever we know about SA - good, done
		pa = copy.deepcopy(self.params)
		pa.decay_schedule = mlrose.GeomDecay()
		pa.random_state = None
		pa.max_iters = 2500
		pa.max_attempts = 250
		self.algorithms.append(SimulatedAnnealing(pa))

		# set up whatever we know about RHC - in progress
		pa = copy.deepcopy(self.params)
		pa.random_state = None
		pa.restarts = 250
		pa.max_iters = 250
		pa.max_attempts = 250
		self.algorithms.append(RandomHillClimbing(pa))

		# set up whatever we know about GA
		pa = copy.deepcopy(self.params)
		pa.pop_size = 1000
		pa.pop_breed_percent = 0.50
		pa.mutation_prob = 0.25
		pa.max_iters = np.inf
		pa.max_attempts = 250
		pa.random_state = None
		self.algorithms.append(GeneticAlgorithm(pa))

		# set up whatever we know about Mimic
		pa = copy.deepcopy(self.params)
		pa.pop_size = 600
		pa.keep_pct = 0.1
		pa.noise = 0.01
		pa.max_iters = np.inf
		pa.max_attempts = 20
		pa.random_state = None
		self.algorithms.append(Mimic(pa))

	@property
	def fitness_label(self):
		return self._fitness_label

	def __str__(self):
		return self.name + "-" + str(self.params.length)