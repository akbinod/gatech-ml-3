import json
import mlrose_hiive as mlrose
import numpy as np
import time
from Solvers import BaseAlgorithm
from akbinod.Utils.TimedFunction import TimedFunction

class Mimic(BaseAlgorithm):
	def __init__(self, params):
		super().__init__(params)
		self.name = "MIMIC"

	@TimedFunction(True)
	def tune(self, problem, init_state, maximizing):
		sols = []
		pop_size = [200, 600, 1000]
		keep_percent = [0.1, 0.25, 0.50]
		noise = [0.0, 0.05, 0.1]
		max_attempts = [50,100,500]
		# pop_size = [200]
		# keep_percent = [0.1]
		# noise = [0.0]
		# max_attempts = [50]

		for ps in pop_size:
			for kp in keep_percent:
				for n in noise:
					for ma in max_attempts:
						sol = {}
						sols.append(sol)
						sol["pop_size"] = ps
						sol["keep_percent"] = kp
						sol["noise"] = n
						sol["max_attempts"] = ma
						proc_time1 = time.process_time()
						_, _ , iteration_scores = mlrose.mimic(
												problem
												, pop_size = ps
												, keep_pct = kp
												, noise = n
												, max_attempts= ma
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

		print(f"pop_size\tkeep_p\tnoise\tmax_att\tfitness\tbest_at\titers\ttime")
		for sol in sols:
			print(f"{sol['pop_size']}\t{sol['keep_percent']}\t{sol['noise']}\t{sol['max_attempts']}\t{sol['best_fitness']}\t{sol['best_score_at']}\t{sol['iterations']}\t{sol['time']}")

		return

	@TimedFunction(True)
	def solve(self, problem, init_state):
		# Solve problem using Genetic Algorithms
		t1 = time.process_time()
		self.best_state, self.best_fitness, self.iteration_scores = mlrose.mimic (
												problem
												, pop_size = self.params.pop_size
												, keep_pct = self.params.keep_pct
												, noise = self.params.noise
												, max_iters = self.params.max_iters
												, max_attempts = self.params.max_attempts
												, random_state=1
												, curve = True)

		t2 = time.process_time()
		self.solve_time = t2 - t1

		return

