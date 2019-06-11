from sunrisePy import sunrisePy
import time
import numpy as np
ip='172.31.1.148'
# ip='localhost'
iiwa=sunrisePy(ip)
iiwa.setBlueOn()
time.sleep(2)
iiwa.setBlueOff()


try:
   while True:
       print(iiwa.getJointsMeasuredTorques())
       time.sleep(0.2)
except KeyboardInterrupt:
    iiwa.close()
    print('an error happened')
    raise
