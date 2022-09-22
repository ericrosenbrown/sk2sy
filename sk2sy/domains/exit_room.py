import math
import random
import copy
import numpy as np

from sk2sy.domains.domain import Domain
from sk2sy.classes import Action, State, Reward

#TODO connect with gym setup
#TODO unit tests

class ExitRoom(Domain):
	# TODO state vector and names vector could be generated in the same place
	# to avoid bugs when we update one but not the other
	state_var_names: list[str] = ["door_loc0", "door_loc1", "switch0_loc0","switch0_loc1", "switch1_loc0", "switch1_loc1", "robot_loc0", "robot_loc1", "door_status", "switch0_status", "switch1_status"]
	def __init__(self):
		'''
		The ExitRoom domain. There is a door the robot must be opened. In order to unlock the door,
		the room must toggle all of the switches on. Once the door is unlocked, the robot can open the door. When robot moves, 
		it will move somewhere within a radius r of the object. A precondition for all non-move actions is to be within radius
		r of the object. 
		'''

		### State ###
		self.reset()

		### Actions ###
		self.actions = [
		"move_to_door",
		"move_to_switch0",
		"move_to_switch1",
		"toggle_on_switch0",
		"toggle_on_switch1",
		"open_door"
		]

		### Dynamics ###
		self.radius = 0.1 #the radius around an object the robot will move / needs to be within to interact with object

	def get_state(self) -> State:
		'''
		Returns the curent state of the environment
		'''
		state = [
			self.door_loc[0],
			self.door_loc[1],
			self.switch0_loc[0],
			self.switch0_loc[1],
			self.switch1_loc[0],
			self.switch1_loc[1],
			self.robot_loc[0],
			self.robot_loc[1],
			float(self.door_status),
			float(self.switch0_status),
			float(self.switch1_status)
		] 
		# Make it a tuple so we it is hashable
		return(State(tuple(state)))

	

	def reset(self) -> State:
		'''
		Reset the state of the environment

		Returns: 
		@initial_state: a list representing the current state (should be initial state from reset)
		'''

		#The position of the doors, switches, and robot
		self.door_loc = [0,0]
		self.switch0_loc = [-1,0]
		self.switch1_loc = [1,0]
		self.robot_loc = [0,0]

		#the status of the door and switches.
		#whether the door is locked or unlocked is implicit in the state of the switch statuses
		self.door_status = False #False means closed, True means open
		self.switch0_status = False #False means off, True means on
		self.switch1_status = False #False means off, True means on

		initial_state = self.get_state()
		return(initial_state)

	def sample_nearby_loc(self, loc: list) -> list:
		'''
		Given a location loc, samples a location nearby_loc that is within radius distance

		Parameters:
		@loc: a list containing the 2D coordinates of the desired location

		Returns:
		@nearby_loc: A list containing the 2D coordinates of location within radius distance to loc
		'''

		#sample random angle
		theta = random.uniform(0,2*math.pi)
		#sample random distance up to radius
		distance = random.uniform(0,self.radius)

		#get x and y displacement
		x_delta = math.cos(theta) * distance
		y_delta = math.sin(theta) * distance

		#add on displacements to loc
		nearby_loc = copy.copy(loc)
		nearby_loc[0] += x_delta
		nearby_loc[1] += y_delta

		return(nearby_loc)

	def is_nearby(self, loc1: list, loc2: list) -> bool:
		'''
		Given two locations loc1 and loc2, returns a boolean determining whether loc1 is within radius distance of loc2

		Parameters:
		@loc1: a list containing the 2D coordinates for location 1
		@loc2: a list containing the 2D coordinates for location 2

		Returns
		@nearby: a boolean determining whether loc1 is with radius distance of loc2 
		'''

		distance = np.linalg.norm(np.array(loc1) - np.array(loc2))
		nearby = (distance <= self.radius)
		return(nearby)

	def step(self, action: Action) -> list[State, Reward, bool]:
		'''
		Given an action, performs action in the environment and returns the current state and reward. If action is infeasible
		in current state, returns an error.

		Parameters:
		@action: a string, representing an action from the actions list.

		Returns:
		@new_state: a state representing the current state resulting from the state
		@reward: float reward value for transition
		@done: boolean representing whether task is complete (door is open)
		'''

		assert action in self.actions

		#perform associated action if preconditions are met
		if action == "move_to_door":
			#no preconditions for moving, get nearby loc and move to it
			nearby_loc = self.sample_nearby_loc(self.door_loc)
			#move robot to that state
			self.robot_loc = nearby_loc
		elif action == "move_to_switch0":
			#no preconditions for moving, get nearby loc and move to it
			nearby_loc = self.sample_nearby_loc(self.switch0_loc)
			#move robot to that state
			self.robot_loc = nearby_loc
		elif action == "move_to_switch1":
			#no preconditions for moving, get nearby loc and move to it
			nearby_loc = self.sample_nearby_loc(self.switch1_loc)
			#move robot to that state
			self.robot_loc = nearby_loc
		elif action == "toggle_on_switch0":
			#switch can only be toggled on if it currently off and the robot is within range
			if self.is_nearby(self.robot_loc, self.switch0_loc) and not self.switch0_status:
				self.switch0_status = True
			else:
				raise Exception("Attempted to perform {action} but preconditions not met".format(action=action))
		elif action == "toggle_on_switch1":
			#switch can only be toggled on if it currently off and the robot is within range
			if self.is_nearby(self.robot_loc, self.switch1_loc) and not self.switch1_status:
				self.switch1_status = True
			else:
				raise Exception("Attempted to perform {action} but preconditions not met".format(action=action))
		elif action == "open_door":
			#door can only be opened if door is closed, door is unlocked (all switches are on) and robot is nearby door
			if self.is_nearby(self.robot_loc, self.door_loc) and self.switch0_status and self.switch1_status and not self.door_status:
				self.door_status = True
			else:
				raise Exception("Attempted to perform {action} but preconditions not met".format(action=action))

		new_state = self.get_state()
		reward = Reward(-1) #all actions have some reward value
		done = self.door_status
		return(new_state, reward, done)


if __name__ == "__main__":
	domain = ExitRoom()

	print(domain.get_state())

	#Test a plan sequence
	plan = [
	"move_to_switch0",
	"toggle_on_switch0",
	"move_to_switch1",
	"toggle_on_switch1",
	"move_to_door",
	"open_door"
	]
	plan = [Action(x) for x in plan]

	for action in plan:
		state, reward, done = domain.step(action)
		print(state, reward, done)