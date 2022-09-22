from collections import defaultdict
import functools
from typing import Optional

from sk2sy.utils.get_mask import get_mask
from sk2sy.utils.generate_transitions import generate_transitions
from sk2sy.utils.partition_by_function import partition_by_function
from sk2sy.utils.invert_dict import invert_dict
from sk2sy.domains.domain import Domain
from sk2sy.classes import Transition, StateVar, State, Action, Factor


#TODO: rewrite compute factors in the same way in original paper, or at least make sure it is in same order of complexity
#TODO: add debugging statements for prints
#TODO: unit tests
#TODO: restructure return to just be a single dictionary? what's the best data structure for holding factors?
#TODO: maybe make a single compute_factors function that simply takes list of transitions (where actions are already assumed to be subgoal)
#TODO: (Compute factors by partioning state variables based on option_mask_dict) in functions may be bugged and needs to be checked/fixed?
#TODO: update the docstrings correctly for these functions

def compute_factors_from_transitions(transitions: list[Transition],
	state_var2name: Optional[list[str]] = None
	) -> tuple[
		list[Factor],
		dict[int, list[Action]]
	]:
	'''
	Compute the factors (parition over state variables based on similar option masks) for a given set of subgoal options.
	This function is used when the domain originally does not have subgoal options, and so one of the partition_options
	functions was used.
	'''

	# Mapping from state_var to list of options affecting it
	# TODO populate
	statevar2options: dict[StateVar, set[Action]] = dict()
	for t in transitions:
		state_len = len(t.start_state)
		affected_vars = [i for i in range(state_len) if t.start_state[i] != t.end_state[i]]
		for v in affected_vars:
			if v not in statevar2options.keys():
				statevar2options[v] = set()
			statevar2options[v].add(t.action)

	# Partition state vars by the set of options affecting them
	# MFNOTE: This currently omits state vars that are never affected
	# To get around that, we'd need to know the names of all state vars,
	# ie how many there are.
	# We could do that in the above loop, but that's gross, so
	# I'll just leave this until the domain has that property

	# MFNOTE: in the pseudocode, we have a list of factors for each option.
	# Here we have a single 

	options2mask: dict[frozenset[Action], list[StateVar]] = partition_by_function(statevar2options.keys(), lambda s: frozenset(statevar2options[s]))
	factors: list[Factor] = []
	factor2options: dict[Factor, frozenset[Action]] = dict()
	for i, (options, state_vars) in enumerate(options2mask.items()):
		if state_var2name is not None:
			state_var_names = tuple([state_var2name[sv] for sv in state_vars])
			f = Factor(str(i), tuple(state_vars), state_var_names=state_var_names)
		else:
			f = Factor(str(i), tuple(state_vars))

		factors.append(f)
		factor2options[f] = options


	# factors: list[Factor] = [Factor(str(i), tuple(vs)) for i, vs in enumerate(options2statevars.values())]
	# # factor2statevars = {i:vs for i, vs in enumerate(options2statevars.values())}
	# factor2options = {i:x for i, x in enumerate(options2statevars.keys())}

	# TODO build option2factors in the above loop and skip factor2options
	option2factors: dict[Action, list[Factor]] = dict()
	for factor, options in factor2options.items():
		for o in options:
			if o not in option2factors.keys():
				option2factors[o] = [factor]
			else:
				option2factors[o].append(factor)

	return factors, option2factors


def compute_factors_domain(domain: Domain, num_transitions:float = 100, state_var2name: Optional[list[str]] = None
	) -> tuple[
		dict[int, list[Factor]],
		dict[int, list[Action]]
	]:
	'''
	Compute the factors (parition over state variables based on similar option masks) for a domain. This implicitly assumes
	the domain already has subgoal options, so this function will sample num_transitions transitions from it to generate
	masks and then factors

	Parameters:
	@domain: a domain to compute the factors for
	@num_transitions (optional): number of transitions to sample from domain to calculate masks for factors.

	Returns:
	@factors_state_idxs: a dictionary, where key is factor_id, and value is state_idxs where state_idxs
	is a list of state_idxs associated with factor_id 
	@factors_options: a dictionary, where key is factor_id, and value is list of options that have mask over
	state_variables included in factors_state_idxs[factor_id]
	'''

	transitions = generate_transitions(domain, num_transitions=num_transitions)
	return(compute_factors_from_transitions(transitions, state_var2name=state_var2name))



def test_compute_factors_domain(num_transitions: int = 100):
	#Tests the compute_factors_domain function 
	from sk2sy.domains.exit_room import ExitRoom
	domain = ExitRoom()
	factors, option2factors = compute_factors_domain(domain, num_transitions, state_var2name=domain.state_var_names)
	factor2options = invert_dict(option2factors)
	print("~" * 100)
	print(f"{len(factors)} factors:")
	for f, options in factor2options.items():
		print(f"Factor: {f}\nOptions: {options}")
		print("")
	# for f in factors:
	# 	print(f)

	# print("factor state idxs {f}".format(f=factors_state_idxs))
	# print("factor options {f}".format(f=factors_options))

def test_compute_factors_from_transitions(num_transitions: int = 100):
	#Tests the compute_factors_from_subgoal_option_transitions function
	from sk2sy.domains.exit_room import ExitRoom
	from sk2sy.algorithms.partition_options import deterministic_partition_options
	domain = ExitRoom()
	#Generate transitions from domain
	transitions = generate_transitions(domain, num_transitions = num_transitions)
	#Get partioned options
	partitioned_options = deterministic_partition_options(transitions)
	#Compute factors
	factors_state_idxs, factors_options = compute_factors_from_transitions(partitioned_options, state_var2name=domain.state_var_names)
	# print("factor state idxs {f}".format(f=factors_state_idxs))
	# print("factor options {f}".format(f=factors_options))



if __name__ == "__main__":
	#how many transitions we sample from domain (For estimating masks, subgoal partitioning
	num_transitions: int = 1000
	test_compute_factors_domain(num_transitions)
