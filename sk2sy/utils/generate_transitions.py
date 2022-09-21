from typing import List
import random

from sk2sy.classes import Transition
from sk2sy.domains.domain import Domain


def generate_transitions(domain: Domain, num_transitions = 100) -> List[Transition]:	
	'''
	Generate num_transitions transitions by stepping through the domain. Takes random actions,
	and once done is returned from domain, resets the domain.
	'''
	transitions = []

	while len(transitions) < num_transitions:
		state = domain.get_state()
		action = random.choice(domain.actions)
		try:
			next_state, reward, done = domain.step(action)
			transitions.append(Transition(state, action, reward, next_state))
			if done:
				domain.reset()
		except:
			pass

	return(transitions)

#TODO allow user to feed in policy so that it's not just random
#TODO standardize this to gym domain setup for domain
#TODO unit tests