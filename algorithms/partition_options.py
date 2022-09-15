from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
from sk2sy.utils.get_mask import get_mask

#TODO turn this into a class, probably. or move some of these functions into a utils.py function if used across other algos
#TODO standardize the form of transitions (probably make a class for transitions)
#TODO enable parameters to be given for clustering
#TODO make prints prettier and optional with debug flag
#TODO make a more interesting domain and test this code base

def partition_options(transitions: list, eps:float = 1e-2, min_samples:float = 5) -> list:
	'''Given a list of transitions, return a new list of transitions with options partitioned
	into approximately strong subgoal options (Pr(s'|s,o) = Pr(s'|o))

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
	actions = list(set(map(lambda transition: transition[1], transitions)))

	#For each action
	for action in actions:
		print("Building partitions for {action}".format(action=action))
		#Get all transitions associated with that action
		action_transitions = list(filter(lambda transition: transition[1] == action, transitions))
		#print(action_transitions)

		#partition each transition based on mask
		masks_transition_dict = defaultdict(lambda : []) #keys are tuples containing state idx representing mask, values are list of transitions that have that mask

		#loop through each transitions and partition transitions based on mask
		for transition in action_transitions:
			#calculate mask for transition
			mask = get_mask(transition[0], transition[3])

			masks_transition_dict[tuple(mask)].append(transition)

		#print("masks dictionary: {m}".format(m=masks_transition_dict))

		#cluster effect states for each mask and create partitions
		mask_partition_transition_dict = defaultdict(lambda :[]) #keys are tuples containing (mask, partition), values are list of transitions with that mask and in that partition
		for mask, transition_list in masks_transition_dict.items():
			#print(mask, transition_list)
			effect_states = list(map(lambda transition: transition[3], transition_list))
			#print("effect states {e}".format(e=effect_states))
			effect_states_np = np.array(effect_states)

			clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(effect_states_np)
			print(clustering.labels_)
			for (transition, cluster) in zip(transition_list, clustering.labels_):
				mask_partition_transition_dict[(mask, cluster)].append(transition)
		print("mask partition_transition_dict {m}".format(m=mask_partition_transition_dict.keys()))

		#TODO Steps 3) and 4) outlined on page 255 of skills to symbols
		#3) for each pair of partitions, merge data into single partition if start state samples substantially overlap
		#4) create an outcome for each effect and assigned probability


if __name__ == "__main__":
	from sk2sy.domains.exit_room import ExitRoom
	import random

	domain = ExitRoom()

	#Generate transitions from domain
	transitions = []

	num_transitions = 100
	while len(transitions) < num_transitions:
		state = domain.get_state()
		action = random.choice(domain.actions)
		try:
			next_state, reward, done = domain.step(action)
			transitions.append([state, action, reward, next_state])
			if done:
				domain.reset()
		except:
			pass

	#Get partioned transitions
	partioned_transitions = partition_options(transitions)