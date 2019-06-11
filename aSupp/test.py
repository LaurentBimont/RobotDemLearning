from sunrisePy import sunrisePy
import time
import numpy as np
ip='172.31.1.148'
#ip='localhost'
iiwa=sunrisePy(ip)
iiwa.setBlueOn()
time.sleep(2)
iiwa.setBlueOff()

def status():
    print('End effector position and orientation is:')
    print(iiwa.getEEFPos())
    time.sleep(0.1)
    print('Forces acting at end effector are')
    print(iiwa.getEEF_Force())
    time.sleep(0.1)
    print('Cartesian position (X,Y,Z) of end effector')
    print(iiwa.getEEFCartesianPosition())
    time.sleep(0.1)
    print('Moment at end effector')
    print(iiwa.getEEF_Moment())
    time.sleep(0.1)
    print('Joints positions')
    print(iiwa.getJointsPos())
    time.sleep(0.1)
    print('External torques at the joints')
    print(iiwa.getJointsExternalTorques())
    time.sleep(0.1)
    print('Measured torques at the joints')
    print(iiwa.getJointsMeasuredTorques())
    time.sleep(0.1)
    print('Measured torque at joint 5')
    print(iiwa.getMeasuredTorqueAtJoint(5))
    time.sleep(0.1)
    print('Rotation of EEF, fixed rotation angles (X,Y,Z)')
    print(iiwa.getEEFCartesianOrientation())
    time.sleep(0.1)

    print('Joints positions has been streamed external torques are:')
    print(iiwa.getEEFCartesianOrientation())
    time.sleep(0.1)

# try:
#     vel = 30
#     p1=iiwa.getEEFPos()
#     print(p1)
#     p1[0]=100

#     a=iiwa.movePTPLineEEF(p1,vel)
#     # time.sleep()
#     a=iiwa.soc.receive()

#     print(iiwa.getEEFPos())
# except Exception or KeyboardInterrupt:
#     iiwa.close()
#     print('an error happened')
#     raise

# finally:
#     iiwa.close()


from datetime import datetime
import math
from math import pi
start_time = datetime.now()
# returns the elapsed seconds since the start of the program


jointHome = [0, 0, 0, -pi/2, 0, pi/2, 0]
relVel = 0.1

iiwa.movePTPJointSpace(jointHome, relVel)

vel = 10
p1 = iiwa.getEEFPos()
p1[0] += 30
print(p1)
iiwa.movePTPLineEEF(p1, vel)
iiwa.close()
