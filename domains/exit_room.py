import math
import random
import copy
import numpy as np

class ExitRoom:
	def __init__(self):
		'''
		The ExitRoom domain. There is a door the robot must walk through to exit the room. In order to unlock the door,
		the room must toggle all of the switches on. Once the door is unlocked, the robot can open the door. When
		the door is open, the robot can walk through the door and exit the room. When robot moves, it will move 
		somewhere within a radius r of the object. A precondition for all non-move actions is to be within radius
		r of the object. 
		'''

		### State ###
		reset()

		### Actions ###
		self.actions = [
		"move_to_door",
		"move_to_switch0",
		"move_to_switch1",
		"toggle_on_switch0",
		"toggle_on_switch1",
		"open_door",
		"exit_room"
		]

		### Dynamics ###
		self.radius = 0.1 #the radius around an object the robot will move / needs to be within to interact with object

	def get_state(self) -> list:
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
			self.door_status,
			self.switch0_status,
			self.switch1_status
		] 
		return(state)


	def reset(self): -> list
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
		self.door_status = 0 #0 means closed, 1 means open
		self.switch0_status = 0 #0 means off, 1 means on
		self.switch1_status = 1 #0 means off, 1 means on

		initial_state = get_state()
		return(initial_state)

	def sample_nearby_loc(self, loc -> list): -> list
		'''
		Given a location loc, samples a location nearby_loc that is within radius distance

		Parameters:
		@loc: a list containing the 2D coordinates of the desired location

		Returns:
		@nearby_loc: A list containing the 2D coordinates of location within radius distance to loc
		'''

		#sample random angle
		theta = random.randrange(0,2*math.pi,0)
		#sample random distance up to radius
		distance = random.randrange(0,self.radius)

		#get x and y displacement
		x_delta = math.cos(theta) * distance
		y_delta = math.sin(theta) * distance

		#add on displacements to loc
		nearby_loc = copy.copy(loc)
		nearby_loc[0] += x_delta
		nearby_loc[1] += y_delta

		return(nearby_loc)

	def is_nearby(self, loc1 -> list, loc2 -> list): -> bool
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

	def step(self, action: str): -> list
		'''
		Given an action, performs action in the environment and returns the current state. If action is infeasible
		in current state, returns an error.

		Parameters:
		@action: a string, representing an action from the actions list.

		Returns:
		@new_state: a state representing the current state resulting from the state
		'''

		assert action in self.actions

		#perform associated action if preconditions are met
		if action == "move_to_door":
			#no preconditions for moving, get nearby loc and move to it
			nearby_loc = sample_nearby_loc(self.door_loc)
			#move robot to that state
			self.robot_loc = nearby_loc
		elif action == "move_to_switch0":
			#no preconditions for moving, get nearby loc and move to it
			nearby_loc = sample_nearby_loc(self.switch0_loc)
			#move robot to that state
			self.robot_loc = nearby_loc
		elif action == "move_to_switch1":
			#no preconditions for moving, get nearby loc and move to it
			nearby_loc = sample_nearby_loc(self.switch1_loc)
			#move robot to that state
			self.robot_loc = nearby_loc
		elif action == "toggle_on_switch0":
			#switch can only be toggled on if it currently off and the robot is within range
			if self.is_nearby(self.robot_loc, self.switch0_loc) and not self.switch0_status:
				pass #TODO fill in
		elif action == "toggle_on_switch1":
			#switch can only be toggled on if it currently off and the robot is within range
			if self.is_nearby(self.robot_loc, self.switch1_loc) and not self.switch1_status:
				pass #TODO fill in
		elif action == "open_door":
			#door can only be opened if door is closed, door is unlocked (all switches are on) and robot is nearby door
			if self.is_nearby(self.robot_loc, self.door_loc) and self.switch0_status and self.switch1_status and not self.door_status:
				pass #TODO fill in
		elif action == "exit_room":
			#room can be exited if door is open and robot is nearby door
			if self.is_nearby(self.robot_loc, self.door_loc) and self.door_status:
				






