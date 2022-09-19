from collections import defaultdict
import functools

from sk2sy.utils.get_mask import get_mask
from sk2sy.utils.generate_transitions import generate_transitions
from sk2sy.utils.partition_by_function import partition_by_function
from sk2sy.utils.invert_dict import invert_dict
from sk2sy.domains.domain import Domain
from sk2sy.transitions import Transition


#TODO: rewrite compute factors in the same way in original paper, or at least make sure it is in same order of complexity
#TODO: add debugging statements for prints
#TODO: unit tests
#TODO: restructure return to just be a single dictionary? what's the best data structure for holding factors?
#TODO: maybe make a single compute_factors function that simply takes list of transitions (where actions are already assumed to be subgoal)
#TODO: (Compute factors by partioning state variables based on option_mask_dict) in functions may be bugged and needs to be checked/fixed?

def compute_factors_from_subgoal_option_transitions(subgoal_options):
	'''
	Compute the factors (parition over state variables based on similar option masks) for a given set of subgoal options.
	This function is used when the domain originally does not have subgoal options, and so one of the partition_options
	functions was used.

	Parameters:
	@subgoal_options: a dictionary, where keys are actions, values are a list of transitions (state,action,reward,next_state)
	that fall into the option partition.

	Returns:
	@factors_state_idxs: a dictionary, where key is factor_id, and value is state_idxs where state_idxs
	is a list of state_idxs associated with factor_id 
	@factors_options: a dictionary, where key is factor_id, and value is list of options that have mask over
	state_variables included in factors_state_idxs[factor_id]
	'''
	factors_state_idxs = {}
	factors_options = defaultdict(lambda : [])

	option_mask_dict = {} #dictionary containing keys for options and values being list of masked state variables

	#for each action, calculate the mask
	for action in list(subgoal_options.keys()):
		transitions: list[Transition] = subgoal_options[action] #transitions associated with this subgoal option
		print("Calculating mask for action {action}".format(action=action))

		#Get the masks for all transitions in action_transitions
		masks = list(map(lambda transition:get_mask(transition.start_state,transition.end_state), transitions))
		print(masks)

		#use reduce to turn list of masks (list of lists) into just a list of state variables, convert to set and back to list
		#to get unique state variables
		option_mask = list(set(functools.reduce(tuple.__add__, masks, tuple())))
		#sort option mask
		option_mask.sort() 
		
		option_mask_dict[action] = tuple(option_mask)

	print("option mask dict {o}".format(o=option_mask_dict))

	#Compute factors by partioning state variables based on option_mask_dict
	for option, mask in option_mask_dict.items():
		if mask not in factors_state_idxs.values(): #a new factor is being made
			factor_id = len(list(factors_state_idxs.keys()))
			factors_state_idxs[factor_id] = mask
		else: #factor already exists that overlaps with this mask, get the factor id associated with mask
			factor_id = list(factors_state_idxs.values()).index(mask)
		#add option to dictionary based on factor_id
		factors_options[factor_id].append(option)

	return(factors_state_idxs,factors_options)

def compute_factors_domain(domain, num_transitions:float = 100):
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
	factors_state_idxs = {}
	factors_options = defaultdict(lambda : [])

	option_mask_dict = {} #dictionary containing keys for options and values being list of masked state variables

	#generate random transitions for the masks
	transitions = generate_transitions(domain, num_transitions=num_transitions)

	#for each action, calculate the mask
	for action in domain.actions:
		print("Calculating mask for action {action}".format(action=action))

		#Get all transitions with the current action we're getting mask for
		action_transitions: list[Transition] = list(filter(lambda transition: transition.action == action, transitions))
		#Get the masks for all transitions in action_transitions
		masks = list(map(lambda transition:get_mask(transition.start_state,transition.end_state), action_transitions))
		print(masks)

		#use reduce to turn list of masks (list of lists) into just a list of state variables, convert to set and back to list
		#to get unique state variables
		option_mask = list(set(functools.reduce(tuple.__add__, masks, tuple())))
		#sort option mask
		option_mask.sort() 
		
		option_mask_dict[action] = tuple(option_mask)

	print("option mask dict {o}".format(o=option_mask_dict))

	#Compute factors by partioning state variables based on option_mask_dict
	for option, mask in option_mask_dict.items():
		if mask not in factors_state_idxs.values(): #a new factor is being made
			factor_id = len(list(factors_state_idxs.keys()))
			factors_state_idxs[factor_id] = mask
		else: #factor already exists that overlaps with this mask, get the factor id associated with mask
			factor_id = list(factors_state_idxs.values()).index(mask)
		#add option to dictionary based on factor_id
		factors_options[factor_id].append(option)

	return(factors_state_idxs,factors_options)


def compute_factors_domain_2(domain: Domain, num_transitions:float = 100) -> tuple[dict, dict]:
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

	# Mapping from state_var to list of options affecting it
	# TODO populate
	statevar2options: dict[int, set[str]] = dict()
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
	options2statevars = partition_by_function(statevar2options.keys(), lambda s: frozenset(statevar2options[s]))
	factor2statevars = {i:vs for i, vs in enumerate(options2statevars.values())}
	factor2options = {i:x for i, x in enumerate(options2statevars.keys())}

	return factor2statevars, factor2options



if __name__ == "__main__":
	#how many transitions we sample from domain (For estimating masks, subgoal partitioning
	num_transitions = 100

	#Tests the compute_factors_domain function 
	from sk2sy.domains.exit_room import ExitRoom
	domain = ExitRoom()
	factors_state_idxs, factors_options = compute_factors_domain(domain, num_transitions)
	print("factor state idxs {f}".format(f=factors_state_idxs))
	print("factor options {f}".format(f=factors_options))

	#Tests the compute_factors_from_subgoal_option_transitions function
	from sk2sy.algorithms.partition_options import deterministic_partition_options
	domain = ExitRoom()
	#Generate transitions from domain
	transitions = generate_transitions(domain, num_transitions = num_transitions)
	#Get partioned options
	partitioned_options = deterministic_partition_options(transitions)
	#Compute factors
	factors_state_idxs, factors_options = compute_factors_from_subgoal_option_transitions(partitioned_options)
	print("factor state idxs {f}".format(f=factors_state_idxs))
	print("factor options {f}".format(f=factors_options))


