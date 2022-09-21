from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import copy
from typing import NewType

from sk2sy.classes import Transition, State, Factor, Action


def concat_lists(*lsts):
	r = []
	for l in lsts:
		r += l
	return r

def project(s: State, s_vars: Factor) -> State:
	idx_keep = [i for i in range(len(s)) if i not in s_vars]
	return s[idx_keep]

#TODO fill in the docstring for generate_symbol_sets fully
def generate_symbol_sets(option2transitions: dict[Action, list[Transition]], option2factors: dict[Action, list[Factor]]):
	'''
	Generate the symbolic vocabulary
	Psuedocode on page 237 (23)

	The pseudocode checks whether we can make a symbol from a 
	factor based on whether
		Project(e, fi) ∩ Project(e, f \ fi) = e

	This works fine when we can enumerate all states, and all of e
	However, when we only have sampled states, we need to use an approximation
	MFNOTE: TODO Explain what goes wrong if we try this directly without approximation

	We will instead train two e classifiers: one based only on fi, 
	and one based only on  \bigcup (f \ fi)

	f\fi is msk_other, and Project(s,f\fi) is the part of s that
	only contains state vars in f


	'''
	symbols = []
	# all_factors = sorted(list(set(concat_lists(option2factors.values()))))
	for o, ts in option2transitions.items():
		# TODO? take in list of states for e and all_states
		# so that we can precompute those and reuse it in multiple
		# places
		# We sort in case we want to reproduce results with a fixed random seed later
		# other_factors = copy.copy(all_factors)

		e = sorted(list(set([tuple(t.end_state) for t in ts])))
		# TODO calcualte all_states outside this loop
		all_states = sorted(list(set(e + [tuple(t.start_state) for t in ts])))
		n_states = len(all_states)
		# MFNOTE: Getting these labels could be done much faster, I think
		state_is_e = [s in e for s in all_states]
		f = option2factors[o]
		other_factors = [x for x in f if x != f]

		for f_i in f:
			msk_other = sorted(list(set(concat_lists(*other_factors))))

			# Project e onto f and f_complement
			# states_f = [s[f] for s in all_states]
			states_f = [project(s, f_i) for s in all_states]
			# states_other = [s[msk_other] for s in all_states]
			states_other = [project(s, msk_other) for s in all_states]


			# Split states into training and holdout sets
			rng = np.random.default_rng()
			# Hyperparam: p_train
			p_train = .8
			n_train = int(np.ceil(n_states * p_train))
			idx_train = rng.choice(n_states, size=n_train, replace=False)
			idx_valid = np.setdiff1d(list(range(n_states)), idx_train)

			# Train classifiers on each projection
			# Hyperparam: Should we de-dedupe each of these sets, since the projection may
			# have made states non-unique?
			# Hyperparam: architecture, and hyperparams of the architecture
			tree_f = DecisionTreeClassifier().fit(states_f[idx_train], state_is_e[idx_train])
			tree_other = DecisionTreeClassifier().fit(states_other[idx_train], state_is_e[idx_train])

			# Check whether the product of classifiers is a good classifier
			y_pred_f = tree_f.predict_proba(states_f[idx_valid])
			y_pred_other = tree_other.predict_proba(states_other[idx_valid])
			# Hyperparam: method of combining classifiers
			y_pred_prod_prob = y_pred_f * y_pred_other
			pred_thresh = .5
			y_pred_prod = [0 if y < pred_thresh else 1 for y in y_pred_prod_prob]

			# Hyperparam: Metric and threshold for whether the combined classifier
			# is good enough for this to be a valid symbol
			score = accuracy_score(state_is_e, y_pred_prod)
			score_thresh = .9
			if score > score_thresh:
				# Add the symbol
				symbols.append(tree_other)
				# Filter e TODO
				e = [project(s, f_i) for s in e]
				other_factors = [x for x in other_factors if x != f_i]

	
	pass


if __name__ == "__main__":
	from sk2sy.domains.exit_room import ExitRoom
	from sk2sy.utils.generate_transitions import generate_transitions
	from sk2sy.algorithms.partition_options import deterministic_partition_options

	domain = ExitRoom()
	num_transitions = 10

	#Generate transitions from domain
	transitions = generate_transitions(domain, num_transitions = num_transitions)

	#Get partioned transitions
	partitioned_options = deterministic_partition_options(transitions)
	print(partitioned_options)