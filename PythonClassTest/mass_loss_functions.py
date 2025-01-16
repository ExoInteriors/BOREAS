from mclass import massloss

def massloss_setM(self):
	self.m += 2.0


#bind the functions to the class
massloss.setM = massloss_setM
