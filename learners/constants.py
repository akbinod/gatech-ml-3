from enum import Enum, auto

class LearnerMode(Enum):
	regression = auto(),
	classification = auto(),
	clustering = auto(),
	silhouette = auto(),
	pca_num_comp = auto(),
	pca = auto(),
	ica_num_comp = auto(),
	ica = auto()

SEED = 0

