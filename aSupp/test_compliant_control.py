from sunrisePy import sunrisePy
import time
import numpy as np
from math import pi
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
start_time = datetime.now()
# returns the elapsed seconds since the start of the program
# p1 = iiwa.getEEFPos()
# p1[0]-=30
 
# vel=20
# iiwa.movePTPLineEEF(p1,vel)
 
def getSecs():
   dt = datetime.now() - start_time
   ms = (dt.days * 24 * 60 * 60 + dt.seconds)  + dt.microseconds / 1000000.0
   return ms
thrs = 0.01

jointHome = [0,0,0,-pi/2,0,pi/2,0]
relVel = 0.3

iiwa.movePTPJointSpace(jointHome,relVel)

print(iiwa.getEEFPos())

try:
    # iiwa.soc.receive()
    counter = 0
    index = 0
    T = 10
    w = 2*np.pi/T
    theta = 0
    theta_max = 1 
    jpos = iiwa.getJointsPos()

    jpos0_6 = jpos[index]
    print(jpos)
    theta0=jpos0_6

    iiwa.realTime_startImpedanceJoints(0., 0, 0, 0, 100, 80, 50)
    t0 = getSecs()
    t = getSecs()
    while np.abs(theta) <= theta_max:
        now=getSecs()
        #print(iiwa.getJointsPos()[index])
        if (now-t) > 0.002:
            jpos = iiwa.getJointsPos()
            t = getSecs()
            counter = counter+1
            torque = iiwa.sendJointsPositionsGetExTorque(jpos) 
            theta = jpos[0]-jpos0_6
            # print("angle {}".format(jpos))
        # if counter%1000:
        #     print("torque {} ".format(torque))
    print("elapsed time".format(t-t0))

    iiwa.realTime_stopImpedanceJoints()
    deltat = getSecs()-t0
    print(deltat)
    iiwa.setBlueOff()
except KeyboardInterrupt:
    iiwa.realTime_stopImpedanceJoints()
               
    iiwa.close()
    print('an error happened')
    raise
finally:
    iiwa.realTime_stopImpedanceJoints()
    iiwa.close()
