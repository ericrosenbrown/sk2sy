#TODO rework this function to take an option instead and return the mask, probably by generating transitions inside this function? or maybe separate function for that

def get_mask(start_state,end_state):
	'''
	Calculate the mask (state indicies that change from start_state to end_state)

	Parameters:
	@start_state: a list containing the starting state values
	@end_state: a list containing the end state values, has same size as start stae

	Returns:
	@mask: a list containing the indicies in the state vector that are changed between start_state and end_state
	'''

	assert len(start_state) == len(end_state)

	mask = []
	for index, (first, second) in enumerate(zip(start_state, end_state)):
		if first != second:
			mask.append(index)
	return(mask)

#TODO write unit tests 