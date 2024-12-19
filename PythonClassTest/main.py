from mclass import massloss
import mass_loss_functions

C = massloss()

print(C.m, C.P)

C.setP()
C.setM()

print(C.m, C.P)


