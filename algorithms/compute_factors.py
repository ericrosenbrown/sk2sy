from sk2sy.utils.get_mask import get_mask
from sk2sy.utils.generate_transitions import generate_transitions

from collections import defaultdict
import functools

#TODO: restructure this function to not take in a domain? maybe directly take in option masks, rather than generate samples?
#TODO: rewrite compute factors in the same way in original paper, or at least make sure it is in same order of complexity
#TODO: add debugging statements for prints
#TODO: unit tests
#TODO: restructure return to just be a single dictionary? what's the best data structure for holding factors?

def compute_factors(domain, num_transitions:float = 100):
	'''
	Compute the factors (parition over state variables based on similar option masks) for a domain

	Parameters:
	@domain: a domain to compute the factors for
	@num_transitions (optional): number of transitions to sample from domain to calculate masks for factors.

	Returns:
	@factors_state_idxs: a dictionary, where key is factor_id, and value is state_idxs where state_idxs
	is a list of state_idxs associated with factor_id 
	@factors_options: a dictionary, where key is factor_id, and value is list of options that have mask over
	state_variables included in factors_state_idxs[factor_id]
	'''

	#number of state variables 
	num_state_variables = len(domain.get_state())

	factors_state_idxs = {}
	factors_options = defaultdict(lambda : [])

	option_mask_dict = {} #dictionary containing keys for options and values being list of masked state variables

	#generate random transitions for the masks
	transitions = generate_transitions(domain, num_transitions=num_transitions)

	#for each action, calculate the mask
	for action in domain.actions:
		print("Calculating mask for action {action}".format(action=action))

		#Get all transitions with the current action we're getting mask for
		action_transitions = list(filter(lambda transition: transition[1] == action, transitions))
		#Get the masks for all transitions in action_transitions
		masks = list(map(lambda transition:get_mask(transition[0],transition[3]), action_transitions))
		print(masks)

		#use reduce to turn list of masks (list of lists) into just a list of state variables, convert to set and back to list
		#to get unique state variables
		option_mask = list(set(functools.reduce(list.__add__, masks, [])))
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

if __name__ == "__main__":
	from sk2sy.domains.exit_room import ExitRoom

	domain = ExitRoom()
	factors_state_idxs, factors_options = compute_factors(domain)
	print("factor state idxs {f}".format(f=factors_state_idxs))
	print("factor options {f}".format(f=factors_options))


