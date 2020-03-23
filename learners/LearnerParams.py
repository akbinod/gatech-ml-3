import json
import os
from akbinod import DecayedParameter

class LearnerParams():
	def __init__(self, learner_mode, learning_target, *, epsilon = None, alpha = 0.01, gamma = 0.99, target_refresh_rate_C = 500, mini_batch_size = 16, data_path = "./data", video_root = "", reward_shaper=None, data_separator=",", cv=10):

		self.learner_mode = learner_mode
		self.learning_target = learning_target

		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		# will be provided later by the concrete learner
		self.learner_name = ""
		self.cv = cv

		self.target_refresh_rate_C = target_refresh_rate_C
		self.mini_batch_size = mini_batch_size
		self.data_path = data_path
		self.video_path = video_root
		self.reward_shapers = []
		if not reward_shaper is None:
			self.reward_shapers.append(reward_shaper)
		self.data_separator = data_separator
		self.cluster_labels = None
		#this may get filled in later
		self.action_map = None


		if not self.epsilon is None:
			if not issubclass(self.epsilon.__class__, DecayedParameter):
				#send me DecayedParameter object, or nothing at all
				raise TypeError()
		else:
			# if we were defaulted on the exploration rate, set something up
			self.epsilon = DecayedParameter(.995, .0001, 2000)

		if self.video_path != "":
			#if the user does not want video, thats fine, but supplied paths have to be directories
			if not os.path.exists(self.video_path) or not os.path.isdir(self.video_path):
				raise Exception(f"The following must be a folder in which to place recording files, and must exist: \n{self.video_path}")
			else:
				# at this point there is a proper file name for the data_path - use that
				ff = os.path.split(self.data_path)
				fname = ff[1]
				self.video_path = os.path.join(self.video_path, fname)

	def validate_files(self):
		# validate and adjust the path - it can be either a csv file, or a model file
		# can be the raw csv file for splitting and training
		thepath = os.path.split(self.data_path)[0]
		thefile = os.path.split(self.data_path)[1].split(".")[0]
		ext = os.path.split(self.data_path)[1].split(".")[1]
		self.train_file = os.path.join(thepath,thefile + ".train")
		self.test_file = os.path.join(thepath,thefile + ".test")
		self.model_file = os.path.join(thepath,thefile + "." + self.learner_name + ".model")
		if ext == "csv":
			self.mode = "raw"
			if not os.path.exists(self.data_path):
				raise Exception(f"The following must be an existing file: \n{self.data_path}")
		elif ext == "train":
			self.mode = "train"
			if not (os.path.exists(self.train_file + ".X") and os.path.exists(self.train_file + ".y")):
				raise Exception(f"Both X and y training files must exist: \n{self.data_path}")
		elif ext == "model":
			self.mode = "infer"
			if not (os.path.exists(self.train_file + ".X") and os.path.exists(self.train_file + ".y") and os.path.exists(self.model_file)):
				raise Exception(f"All three - X and y testing files, and the model files must exist: \n{self.data_path}")
		else:
			raise Exception(f"The file must be of type csv, train, or model: \n{self.data_path}")

		return True

	def add_reward_shaper(self, reward_shaper):
		self.reward_shapers.append(reward_shaper)

	def to_json(self):
		res = {}
		res["env_name"] = self.env_name
		res["alpha"] =  str(self.alpha)
		res["gamma"] = str(self.gamma)
		res["epsilon"] = str(self.epsilon)
		res["batch_size"] = self.mini_batch_size
		res["target_refresh_rate_C"] = self.target_refresh_rate_C
		res["data_path"] = os.path.abspath(self.data_path)
		res["video_path"] = os.path.abspath(self.data_path)
		res["reward_shapers"] = [str(rs) for rs in self.reward_shapers]
		res["action_map"] = self.action_map

		return res
