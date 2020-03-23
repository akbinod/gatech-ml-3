import numpy as np
from matplotlib import pyplot as plt
from akbinod.Utils.TimedFunction import TimedFunction

class BaseSolver():
	def __init__(self, params):
		self.params = params

		self.current_fitness_score = None
		self.algorithms = []
		self.plot_colors = ['red', 'blue', 'green', 'black']
		self.sa_params = None
		self.rhc_params = None
		self.ga_params = None
		self.mimic_params = None

	@TimedFunction(True)
	def solve(self, runs = 1):

		self.results = []

		for alg in self.algorithms:
			# so that the fitness function records to the correct array
			scores = []
			times = []
			iterations = []
			function_calls = []

			self.current_fitness_score = alg.fitness_scores
			for i in range(runs):
				# clear out that list
				alg.fitness_scores.clear()
				alg.solve(self.problem, self.init_state)
				if not self.maximize:
					alg.fitness_scores = [- score for i, score in enumerate(alg.fitness_scores)]
					alg.iteration_scores = [- score for i, score in enumerate(alg.iteration_scores)]
				scores.extend(alg.iteration_scores)
				iterations.append(len(alg.iteration_scores))
				function_calls.append(len(alg.fitness_scores))
				times.append(alg.solve_time)

			r = {}
			r["alg"] = str(alg)
			r["time"] = round(float(np.mean(times)),4)
			r["best_fitness"] = max(scores)
			r["average_score"] = round(float(np.mean(scores)),4)
			r["iterations"] = round(float(np.mean(iterations)),0)
			r["fit_fn_calls"] = round(float(np.mean(function_calls)),0)
			self.results.append(r)

		print(f"alg\tbest_f\tavg_f\titers\tfn_cal\ttime")
		for r in self.results:
			print(f"{r['alg']}\t{r['best_fitness']}\t{r['average_score']}\t{r['iterations']}\t{r['fit_fn_calls']}\t{r['time']}")
		print(f"Averages over {runs} run(s).")

		return

	@TimedFunction(True)
	def tune(self):
		for alg in self.algorithms:
			# so that the fitness function records to the correct array
			self.current_fitness_score = alg.fitness_scores
			alg.tune(self.problem, self.init_state, self.maximize)

		return

	def fitness_fn(self, state):

		# delegate to the real fitness function
		ret =  self.fitness_delegate.evaluate(state)
		# record the score for reporting
		self.current_fitness_score.append(ret)
		# to be consumed by mlrose
		return ret

	@property
	def fitness_label(self):
		raise NotImplementedError()

	def __str__(self):
		return self.name

	def plot_comparisons(self):
		_, axes = plt.subplots(1, 2, figsize=(20, 5))

		# Plot fitness over iterations
		axes[0].set_title("Benchmarking Fitness (iterations): " + str(self))
		axes[0].set_xlabel("iterations")
		axes[0].set_ylabel(self.fitness_label)

		for i, alg in enumerate(self.algorithms):
			if len(alg.iteration_scores):
				axes[0].plot(alg.iteration_scores, '-'
							, color=self.plot_colors[i]
							, label= str(alg) +  ": " + str(alg.best_fitness)
							)
		axes[0].legend(loc="best")

		# Plot fitness over fitness function calls
		axes[1].set_title("Benchmarking Fitness (fn calls): " + str(self))
		axes[1].set_xlabel("fitness function calls")
		axes[1].set_ylabel(self.fitness_label)
		for i, alg in enumerate(self.algorithms):
			if len(alg.fitness_scores):
				axes[1].plot(alg.fitness_scores, '-'
							, color=self.plot_colors[i]
							, label= str(alg) +  ": " + str(alg.best_fitness)
							)
		axes[1].legend(loc="best")


		return plt
