from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from functools import reduce
from typing import NewType, Union

from sk2sy.classes import Transition, State, Factor, Action, StateVar, Symbol


def concat_lists(*lsts):
	r = []
	for l in lsts:
		r += l
	return r

def project(s: State, s_vars: tuple[StateVar,...]) -> State:
	idx_keep = [i for i in range(len(s)) if i not in s_vars]
	s_array = np.array(s)
	return State(tuple(s_array[idx_keep]))

#TODO fill in the docstring for generate_symbol_sets fully
def generate_symbol_sets(transitions: list[Transition], option2factors: dict[Action, list[Factor]], verbose: int=0) -> list[Symbol]:
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
	option2transitions: dict[str, list[Transition]] = partition_by_function(transitions, lambda t: t.action)
	symbols = []
	# all_states: list[State]
	#state_pairs = [[t.start_state, t.end_state] for t in transitions]
	# all_states = reduce(lambda t1,t2: [t1.start_state,t1.end_state] + [t2.start_state, t2.end_state], transitions)
	all_states = sorted(list(set(reduce(list.__add__, [[t.start_state, t.end_state] for t in transitions], list()))))
	# all_factors = sorted(list(set(concat_lists(option2factors.values()))))
	for o, ts in option2transitions.items():
		# TODO? take in list of states for e and all_states
		# so that we can precompute those and reuse it in multiple
		# places
		# We sort in case we want to reproduce results with a fixed random seed later
		# other_factors = copy.copy(all_factors)

		e = sorted(list(set([tuple(t.end_state) for t in ts])))
		# TODO calcualte all_states outside this loop
		# all_states = sorted(list(set(e + [tuple(t.start_state) for t in ts])))
		n_states = len(all_states)
		# MFNOTE: Getting these labels could be done much faster, I think
		state_is_e = np.array([s in e for s in all_states])
		f = option2factors[o]
		other_factors: list[Factor] = [x for x in f if x != f]
		if "move" in o:
			1 == 1
		for f_i in f:
			# TODO use reduce
			msk_i = f_i.state_vars
			other_factors_inner: list[Factor] = [f_j for f_j in other_factors if f_j != f_i]
			msk_other = sorted(list(set(concat_lists(*other_factors_inner))))

			# Project e onto f and f_complement
			# states_f = [s[f] for s in all_states]
			states_f = np.array([np.array(project(s, msk_i)) for s in all_states])
			# states_other = [s[msk_other] for s in all_states]
			states_other = np.array([np.array(project(s, msk_other)) for s in all_states])


			# Split states into training and holdout sets
			# Stratify by whether the state is in e
			# rng = np.random.default_rng()
			# Hyperparam: p_train
			p_train = .8
			idx_train, idx_valid = train_test_split(list(range(n_states)), train_size=p_train, stratify=state_is_e)


			# Train classifiers on each projection
			# Hyperparam: Should we de-dedupe each of these sets, since the projection may
			# have made states non-unique?
			# Hyperparam: architecture, and hyperparams of the architecture
			tree_f = DecisionTreeClassifier().fit(states_f[idx_train], state_is_e[idx_train])
			tree_other = DecisionTreeClassifier().fit(states_other[idx_train], state_is_e[idx_train])
			symbol_f = Symbol(tree_other, o, (f_i,))
			symbol_other = Symbol(tree_f, o, tuple(other_factors_inner))

			# Check whether the product of classifiers is a good classifier
			# TODO BUG if we 
			y_pred_prob_f = tree_f.predict_proba(states_f[idx_valid])[:,1]
			y_pred_prob_other = tree_other.predict_proba(states_other[idx_valid])[:,1]
			# Hyperparam: method of combining classifiers
			y_pred_combined_prob = y_pred_prob_f * y_pred_prob_other
			pred_thresh = .5
			if verbose > 1: print(y_pred_combined_prob)
			y_pred_combined = [0 if y < pred_thresh else 1 for y in y_pred_combined_prob]
			# These preds/scores are used for debugging only
			y_pred_f = [0 if y < pred_thresh else 1 for y in y_pred_prob_f]
			y_pred_other = [0 if y < pred_thresh else 1 for y in y_pred_prob_other]

			# Hyperparam: Metric and threshold for whether the combined classifier
			# is good enough for this to be a valid symbol
			score = accuracy_score(state_is_e[idx_valid], y_pred_combined)
			score_f = accuracy_score(state_is_e[idx_valid], y_pred_f)
			score_other = accuracy_score(state_is_e[idx_valid], y_pred_other)
			scores_d = {"f":score_f, "other":score_other, "comb": score}


			score_thresh = .9
			if score > score_thresh:
				# Remove this factor from our factor list
				other_factors = other_factors_inner

				# Add the symbol
				# NOTE: The factors are the ones whose state_vars we care about for this symbol
				# NOT the factors we project away.
				symbols.append(symbol_f)
				# Filter e
				e = [project(s, f_i.state_vars) for s in e]

		# TODO Project out all combinations of remaining factors.
	return symbols

def pretty_print_option2factors(d: dict[Action, list[Factor]]):
	for o, fs in d.items():
		print(o)
		print(fs)
		print("")

if __name__ == "__main__":
	from sk2sy.domains.exit_room import ExitRoom
	from sk2sy.utils.generate_transitions import generate_transitions
	from sk2sy.algorithms.partition_options import deterministic_partition_options
	from sk2sy.algorithms.compute_factors import compute_factors_from_transitions
	from sk2sy.utils.partition_by_function import partition_by_function

	domain = ExitRoom()
	num_transitions = 1000

	#Generate transitions from domain
	transitions = generate_transitions(domain, num_transitions = num_transitions)

	#Get partioned transitions
	subgoal_option_transitions = deterministic_partition_options(transitions)
	# print(subgoal_option_transitions)

	factors, option2factors = compute_factors_from_transitions(subgoal_option_transitions, state_var2name=domain.state_var_names)
	
	pretty_print_option2factors(option2factors)

	symbols = generate_symbol_sets(subgoal_option_transitions, option2factors)
	print(f"# Symbols: {len(symbols)}")
	for s in symbols:
		print(s.option)
		print(s.state_var_names)
		# print(s.state_vars)
		print("")
