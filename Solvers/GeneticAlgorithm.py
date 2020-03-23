import json
import mlrose_hiive as mlrose
import numpy as np
import time
from Solvers import BaseAlgorithm
from akbinod.Utils.TimedFunction import TimedFunction

class GeneticAlgorithm(BaseAlgorithm):
	def __init__(self, params):
		super().__init__(params)
		self.name = "GA"

	@TimedFunction(True)
	def tune(self, problem, init_state, maximizing):
		sols = []
		pop_size = [20, 200, 1000]
		pop_breed_percent = [0.1, 0.25, 0.50]
		mutation_prob = [0.10, 0.25, 0.50]

		for ps in pop_size:
			for pbp in pop_breed_percent:
				for mp in mutation_prob:
					sol = {}
					sols.append(sol)
					sol["pop_size"] = ps
					sol["pop_breed_percent"] = pbp
					sol["mutation_prob"] = mp

					proc_time1 = time.process_time()
					_, _ , iteration_scores = mlrose.genetic_alg(
												problem
												, pop_size = ps
												, pop_breed_percent= pbp
												, mutation_prob=mp
												, max_attempts = 250
												, max_iters = np.inf
												, random_state=1
												, curve = True
												)
					proc_time2 = time.process_time()
					sol["time"] = round(proc_time2 - proc_time1,5)
					if not maximizing:
						iteration_scores = [-score for i, score in enumerate(iteration_scores)]

					sol["best_fitness"] = np.max(iteration_scores)
					sol["best_score_at"] = float(np.argmax(iteration_scores))
					sol["iterations"] = len(iteration_scores)

		print(f"pop_size\tp_br_p\tmut_pr\tfitness\tbest_at\titers\ttime")
		for sol in sols:
			print(f"{sol['pop_size']}\t{sol['pop_breed_percent']}\t{sol['mutation_prob']}\t{sol['best_fitness']}\t{sol['best_score_at']}\t{sol['iterations']}\t{sol['time']}")

		return

	@TimedFunction(True)
	def solve(self, problem, init_state):
		# Solve problem using Genetic Algorithms
		t1 = time.process_time()
		self.best_state, self.best_fitness, self.iteration_scores = mlrose.genetic_alg(
												problem
												, pop_size = self.params.pop_size
												, pop_breed_percent= self.params.pop_breed_percent
												, mutation_prob=self.params.mutation_prob
												, max_attempts = self.params.max_attempts
												, max_iters = self.params.max_iters
												, random_state=self.params.random_state
												, curve = True
												)
		t2 = time.process_time()
		self.solve_time = t2 - t1

		return

