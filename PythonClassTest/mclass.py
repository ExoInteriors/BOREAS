class massloss:

	"""Class for mass loss project"""

	#list here all parameters with default values
	m = 0.0
	P = 0.0


	def __init__(self):
		#eventually add here a function to read parameters from a file
		#otherwise just set the values here
		self.m = 1.0
		self.P = 2.0

	def setP(self):
		self.P += 1.0

	def setm(self):
		#set pass to move the implementation to another file
		pass


