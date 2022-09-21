from sk2sy.transitions import Transition

#TODO fill in the docstring for generate_symbol_sets fully
def generate_symbol_sets(option2transitions: dict[str, list[Transition]], option2factors: dict[str, list[int]]):
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



	'''
	for o, ts in option2transitions.items():
		e = [t.end_state for t in ts]
		f = option2factors[o]



	raise NotImplementedError()
	pass