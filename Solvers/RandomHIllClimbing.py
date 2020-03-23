import json
import mlrose_hiive as mlrose
import numpy as np
import time
from Solvers import BaseAlgorithm
from akbinod.Utils.TimedFunction import TimedFunction

class RandomHillClimbing(BaseAlgorithm):
	def __init__(self, params):
		super().__init__(params)
		self.name = "RHC"

	@TimedFunction(True)
	def tune(self, problem, init_state, maximizing):

		sols = []

		attempts = [25, 250, 2500]
		iters = [25, 250, 2500, np.inf]
		restarts = [10, 25, 250]
		rando = [None, 1]

		for att in attempts:
			for iter in iters:
				for restart in restarts:
					for ra in rando:
						sol = {}
						sols.append(sol)
						sol["attempts"] = att
						sol["iters"] = iter
						sol["restarts"] = restart
						sol["random_state"] = ra

						proc_time1 = time.process_time()
						_, _ , iteration_scores = mlrose.random_hill_climb(
														problem
														, restarts = restart
														, max_attempts = att
														, max_iters = iter
														, init_state = init_state
														, curve = True
														, random_state=ra)
						proc_time2 = time.process_time()
						sol["time"] = round(proc_time2 - proc_time1,5)
						if not maximizing:
							iteration_scores = [-score for i, score in enumerate(iteration_scores)]

						sol["best_fitness"] = np.max(iteration_scores)
						sol["best_score_at"] = float(np.argmax(iteration_scores))
						sol["iterations"] = len(iteration_scores)
		# print(json.dumps(sols))
		print(f"max_att\tmax_itr\trestrt\trand\tfitness\tbest_at\titers\ttime")
		for sol in sols:
			print(f"{sol['attempts']}\t{sol['iters']}\t{sol['restarts']}\t{sol['random_state']}\t{sol['best_fitness']}\t{sol['best_score_at']}\t{sol['iterations']}\t{sol['time']}")

		return

	@TimedFunction(True)
	def solve(self, problem, init_state):
		# Solve problem using Random Hill Climbing
		t1 = time.process_time()
		self.best_state , self.best_fitness, self.iteration_scores = mlrose.random_hill_climb(
														problem
														, restarts = self.params.restarts
                                                      	, max_attempts = self.params.max_attempts
														, max_iters = self.params.max_iters
                                                      	, init_state = init_state
														, curve = True
														, random_state=self.params.random_state)

		t2 = time.process_time()
		self.solve_time = t2 - t1

		return

