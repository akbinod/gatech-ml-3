import json
import mlrose_hiive as mlrose
import numpy as np
import time
from Solvers.BaseAlgorithm import BaseAlgorithm
from akbinod.Utils.TimedFunction import TimedFunction

class SimulatedAnnealing(BaseAlgorithm):
	def __init__(self, params):
		super().__init__(params)
		self.name = "SA"

	def tune_scheds(self, problem, init_state, maximizing):

		sols = {}
		sols["exp"]= sol = {}
		_, sol["best_fitness"], iteration_scores = mlrose.simulated_annealing(
													problem
													, schedule = mlrose.ExpDecay()
													, max_attempts = 1000
													, max_iters = np.inf
													, init_state = init_state
													, curve = True
													, random_state=1)
		if not maximizing:
			iteration_scores = [- score for i, score in enumerate(iteration_scores)]
		sol["best_score_at"] = float(np.argmax(iteration_scores))
		sol["iterations"] = len(iteration_scores)

		sols["arith"]= sol = {}
		_, sol["best_fitness"], iteration_scores = mlrose.simulated_annealing(
													problem
													, schedule = mlrose.ArithDecay()
													, max_attempts = 1000
													, max_iters = np.inf
													, init_state = init_state
													, curve = True
													, random_state=1)
		if not maximizing:
			iteration_scores = [- score for i, score in enumerate(iteration_scores)]
		sol["best_score_at"] = float(np.argmax(iteration_scores))
		sol["iterations"] = len(iteration_scores)

		sols["geo,"]= sol = {}
		_, sol["best_fitness"], iteration_scores = mlrose.simulated_annealing(
													problem
													, schedule = mlrose.GeomDecay()
													, max_attempts = 1000
													, max_iters = np.inf
													, init_state = init_state
													, curve = True
													, random_state=1)
		if not maximizing:
			iteration_scores = [- score for i, score in enumerate(iteration_scores)]
		sol["best_score_at"] = float(np.argmax(iteration_scores))
		sol["iterations"] = len(iteration_scores)


		# best_sched_name = ""
		# if maximizing:
		# 	best_fitness = 0
		# else:
		# 	best_fitness = np.inf
		# best_iterations = np.inf
		# best_score_at = 0
		# for key in sols:
		# 	sol = sols[key]
		# 	if sol["best_fitness"] >  best_fitness:
		# 		best_fitness = sol["best_fitness"]
		# 		if sol["iterations"] < best_iterations:
		# 			best_iterations = sol["iterations"]
		# 			best_sched_name = key
		# 			best_score_at = sol["best_score_at"]
		# print(best_sched_name, best_fitness, best_iterations, best_score_at)

	def tune_rest(self, problem, init_state, maximizing):
		# for Queens - GeomDecay did best
		sols = []

		attempts = [25, 250, 2500]
		iters = [25, 250, 2500, np.inf]
		rando = [None, 1]

		for att in attempts:
			for iter in iters:
				for ra in rando:
					sol = {}
					sols.append(sol)
					sol["attempts"] = att
					sol["iters"] = iter
					sol["random_state"] = ra

					proc_time1 = time.process_time()
					_, _ , iteration_scores = mlrose.simulated_annealing(
													problem
													, schedule = mlrose.GeomDecay()
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

		print(f"max_att\tmax_itr\trand\tfitness\tbest_at\titers\ttime")
		for sol in sols:
			print(f"{sol['attempts']}\t{sol['iters']}\t{sol['random_state']}\t{sol['best_fitness']}\t{sol['best_score_at']}\t{sol['iterations']}\t{sol['time']}")

		return

	@TimedFunction(True)
	def tune(self, problem, init_state, maximizing):
		# Figure out the best schedule - the answer is GeomSched
		# self.tune_scheds(problem, init_state, maximizing)
		self.tune_rest(problem, init_state, maximizing)

	@TimedFunction(True)
	def solve(self, problem, init_state):
		# Solve problem using simulated annealing

		t1 = time.process_time()
		self.best_state , self.best_fitness, self.iteration_scores = mlrose.simulated_annealing(
														problem
														, schedule = self.params.decay_schedule
                                                      	, max_attempts = self.params.max_attempts
														, max_iters = self.params.max_iters
                                                      	, init_state = init_state
														, curve = True
														, random_state=self.params.random_state)

		t2 = time.process_time()
		self.solve_time = t2 - t1


		return

