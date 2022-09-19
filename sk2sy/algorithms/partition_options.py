from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
from sk2sy.utils.get_mask import get_mask
from sk2sy.transitions import Transition

#TODO turn this into a class, probably. or move some of these functions into a utils.py function if used across other algos
#TODO standardize the form of transitions (probably make a class for transitions)
#TODO make prints prettier and optional with debug flag
#TODO make a more interesting domain and test this code base
#TODO maybe have a better data structure for what's returned from *_partition_options
#TODO partitioned option names have ^ to designate them, make sure actions don't have that and maybe do something smarter later on?
#TODO finished probablistic_partition_options


def deterministic_partition_options(transitions: list[Transition], eps:float = 1e-2, min_samples:float = 5) -> list:
	'''Given a list of transitions, return a new list of transitions with options partitioned
	into approximately strong subgoal options (Im(o,x) = Eff(o) for all x in I_o). This assumes 
	non-probablistic effect distriubtion. Uses DBSCAN for clustering from sklearn implementation

	DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

	Paramters:
	@transitions: a list of (state, action, reward, next_state) transitions from an environment.
	state and next_state are a vector (represented by a list) of same dimension. action is a string
	representing the action, and reward is a float
	@eps (optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
	This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to
	choose appropriately for your data set and distance function.
	@min_samples (optional): The number of samples (or total weight) in a neighborhood for a point to be considered as 
	a core point. This includes the point itself.

	Returns:
	@partioned_options: a dictionary, where keys are named partioned actions, and whose
	values are a list of transitions (same as in @transitions) that fall into the option partition.
	partitioned action names are given by action_masknumber_partition. action is the name of the action
	this partitioned action comes from, masknumber is a unique number representing a particular mask this
	partitioned option has, and partition represents a particular clustering id.
	'''

	partitioned_options = defaultdict(lambda :[]) 

	#Get all the actions in the transitions
	actions = list(set(map(lambda transition: transition.action, transitions)))

	#For each action
	for action in actions:
		#print("Building partitions for {action}".format(action=action))
		#Get all transitions associated with that action
		action_transitions: list[Transition] = list(filter(lambda transition: transition.action == action, transitions))
		#print(action_transitions)

		#partition each transition based on mask
		masks_transition_dict = defaultdict(lambda : []) #keys are tuples containing state idx representing mask, values are list of transitions that have that mask

		#loop through each transitions and partition transitions based on mask
		for transition in action_transitions:
			#calculate mask for transition
			mask = get_mask(transition.start_state, transition.end_state)

			masks_transition_dict[tuple(mask)].append(transition)

		#print("masks dictionary: {m}".format(m=masks_transition_dict))

		#cluster effect states for each mask and create partitions
		for mask_idx, (mask, transition_list) in enumerate(masks_transition_dict.items()):
			transition_list: list[Transition]
			#print(mask, transition_list)
			effect_states = list(map(lambda transition: transition.end_state, transition_list))
			#print("effect states {e}".format(e=effect_states))
			effect_states_np = np.array(effect_states)

			clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(effect_states_np)
			#print(clustering.labels_)
			for (transition, cluster) in zip(transition_list, clustering.labels_):
				partitioned_options[action+"^"+str(mask_idx)+"^"+str(cluster)].append(transition)
	return(partitioned_options)


def probablistic_partition_options(transitions: list[Transition], eps:float = 1e-2, min_samples:float = 5) -> list:
	#TODO: correct docstring and flesh this out fully for probabilistic setting
	#TODO: can I borrow parts of detereministic_partition_options code?
	'''Given a list of transitions, return a new list of transitions with options partitioned
	into approximately strong subgoal options (Pr(s'|s,o) = Pr(s'|o)). Assumes that effect sets
	may be probabilistic, and therefore assigns outcome probabilities to different effects
	for each parttion

	Paramters:
	@transitions: a list of (state, action, reward, next_state) transitions from an environment.
	state and next_state are a vector (represented by a list) of same dimension. action is a string
	representing the action, and reward is a float

	Returns:
	@partioned_transitions: a list of (state, action, reward, next_state, partition) transitions from the environment.
	(state, action, reward, next_state) are the same as in @transitions, partition is an integer determining what the
	partition label associated with this action partion is.
	'''

	#Get all the actions in the transitions
	actions = list(set(map(lambda transition: transition.action, transitions)))

	#For each action
	for action in actions:
		print("Building partitions for {action}".format(action=action))
		#Get all transitions associated with that action
		action_transitions = list(filter(lambda transition: transition.action == action, transitions))
		action_transitions: list[Transition]
		#print(action_transitions)

		#partition each transition based on mask
		masks_transition_dict = defaultdict(lambda : []) #keys are tuples containing state idx representing mask, values are list of transitions that have that mask

		#loop through each transitions and partition transitions based on mask
		for transition in action_transitions:
			#calculate mask for transition
			mask = get_mask(transition.start_state, transition.end_state)

			masks_transition_dict[tuple(mask)].append(transition)

		#print("masks dictionary: {m}".format(m=masks_transition_dict))

		#cluster effect states for each mask and create partitions
		mask_partition_transition_dict = defaultdict(lambda :[]) #keys are tuples containing (mask, partition), values are list of transitions with that mask and in that partition
		for mask, transition_list in masks_transition_dict.items():
			masks_transition_dict: list[Transition]
			#print(mask, transition_list)
			effect_states = list(map(lambda transition: transition.end_state, transition_list))
			#print("effect states {e}".format(e=effect_states))
			effect_states_np = np.array(effect_states)

			clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(effect_states_np)
			print(clustering.labels_)
			for (transition, cluster) in zip(transition_list, clustering.labels_):
				mask_partition_transition_dict[(mask, cluster)].append(transition)
		print("mask partition_transition_dict {m}".format(m=mask_partition_transition_dict.keys()))

		#for each pair of partitions, merge data into single partition if start state samples substantially overlap
		merged_partitions = {} #a dictionary, where keys are 
		if len(mask_partition_transition_dict.keys()) != 1: #if there are more than one partition
			pass #TODO: handle the case when partitions may have to be merged

		#TODO: 4) create an outcome for each effect and assigned probability



if __name__ == "__main__":
	from sk2sy.domains.exit_room import ExitRoom
	from sk2sy.utils.generate_transitions import generate_transitions

	domain = ExitRoom()
	num_transitions = 10

	#Generate transitions from domain
	transitions = generate_transitions(domain, num_transitions = num_transitions)

	#Get partioned transitions
	partitioned_options = deterministic_partition_options(transitions)
	print(partitioned_options)