from iiwaPy.mySock import mySock
from iiwaPy.Getters import Getters
from iiwaPy.Setters import Setters
from iiwaPy.RealTime import RealTime
from iiwaPy.Senders import Senders
from iiwaPy.PTP import PTP
from iiwaPy.check import check_size, checkAcknowledgment 

class sunrisePy:
    getters=0
    realtime=0
    gnerealPorpuse=0
    
    def __init__(self,ip):
        port=30001
        self.soc=mySock((ip,port))
        self.set=Setters(self.soc)
        self.get=Getters(self.soc)
        self.sender=Senders(self.soc)
        self.rtl=RealTime(self.soc)
        self.ptp=PTP(self.soc)
  
    def close(self):
        self.soc.close()
    
    def send(self,data):
        self.soc.send(data+b"\n")
        msg=self.soc.receive()
        if msg == "nak":
            raise ValueError("Received msg is nak, there was a limit violation : watch iiwa log screen for more information")
        return msg

    def __createCommand(self,name, data ):
        command_list = [name]+list(map(str,data))+ [""]
        command = "_".join(command_list).encode("ascii")
        return command

    def __blockUntilAcknowledgment(self):
        while(True):
            msg = self.mysoc.receive()
            if (checkAcknowledgment(msg)):
                break


    def attachToolToFlange(self, ts):
        check_size(6,"Flange frame" ,ts)

        command=self.__createCommand("TFtrans", ts)
        msg=self.send(command)
        print(msg)
        checkAcknowledgment(msg) 
    # PTP motion
    """
    Joint space motion
    """
    def movePTPJointSpace(self,jpos,relVel):
        self.ptp.movePTPJointSpace(jpos,relVel)
    
    def movePTPHomeJointSpace(self,relVel):
        self.ptp.movePTPHomeJointSpace(relVel)
        
    def movePTPTransportPositionJointSpace(self,relVel):
        self.ptp.movePTPTransportPositionJointSpace(relVel)
    """
    Cartesian linear  motion
    """        
    def movePTPLineEEF(self,pos,vel, orientationVel):
        self.ptp.movePTPLineEEF(pos,vel, orientationVel)
        
    def movePTPLineEEFRelBase(self,pos,vel, orientationVel):
        self.ptp.movePTPLineEEFRelBase(pos,vel, orientationVel)
        
    def movePTPLineEEFRelEEF(self,pos,vel, orientationVel):
        self.ptp.movePTPLineEEFRelEEF(pos,vel, orientationVel)
    """
    Circular motion
    """        
    def movePTPCirc1OreintationInter(self, f1,f2, vel):
        self.ptp.movePTPCirc1OrientationInter(f1,f2, vel)
        
    def movePTPArcYZ_AC(self,theta,c,vel):
        self.ptp.movePTPArcYZ_AC(theta,c,vel)
        
    def movePTPArcXZ_AC(self,theta,c,vel):
        self.ptp.movePTPArcXZ_AC(theta,c,vel)
        
    def movePTPArcXY_AC(self,theta,c,vel):
        self.ptp.movePTPArcXY_AC(theta,c,vel)
        
    def movePTPArc_AC(self,theta,c,k,vel):
        self.ptp.movePTPArc_AC(theta,c,k,vel)
        
# realtime motion control
    def realTime_stopImpedanceJoints(self):
        self.rtl.realTime_stopImpedanceJoints()
        
    def realTime_stopDirectServoJoints(self):
        self.rtl.realTime_stopDirectServoJoints()
        
    def realTime_startDirectServoJoints(self):  
        self.rtl.realTime_startDirectServoJoints()
    
    def realTime_startImpedanceJoints(self,weightOfTool,cOMx,cOMy,cOMz,cStiness,rStifness,nStifness):
        self.rtl.realTime_startImpedanceJoints(weightOfTool,cOMx,cOMy,cOMz,cStiness,rStifness,nStifness)
    
    def sendJointsPositions(self,x):
        self.sender.sendJointsPositions(x)
        
    def sendJointsPositionsGetMTorque(self,x): 
        return self.sender.sendJointsPositionsGetMTorque(x)
        
    def sendJointsPositionsGetExTorque(self,x):
        return self.sender.sendJointsPositionsGetExTorque(x)
        
    def sendJointsPositionsGetActualJpos(self,x):
        return self.sender.sendJointsPositionsGetActualJpos(x)
        
# getters
    def getEEFPos(self):
        return self.get.getEEFPos()
    
    def getEEF_Force(self):
        return self.get.getEEF_Force()
        
    def getEEFCartesianPosition(self):
        return self.get.getEEFCartesianPosition()
        
    def getEEF_Moment(self):
        return self.get.getEEF_Moment()
        
    def getJointsPos(self):
        return self.get.getJointsPos()
        
    def getJointsExternalTorques(self):
        return self.get.getJointsExternalTorques()
        
    def getJointsMeasuredTorques(self):
        return self.get.getJointsMeasuredTorques()
        
    def getMeasuredTorqueAtJoint(self,x):
        return self.get.getMeasuredTorqueAtJoint(x)
        
    def getEEFCartesianOrientation(self):
        return self.get.getEEFCartesianOrientation()
        
# get pin states 
    def getPin3State(self):
        return self.get.getPin3State()
    
    def getPin10State(self):
        return self.get.getPin10State()
    
    def getPin13State(self):
        return self.get.getPin13State()
        
    def getPin16State(self):
        return self.get.getPin16State()
        
# setters
    def setBlueOff(self):
        self.set.setBlueOff()
        
    def setBlueOn(self):
        self.set.setBlueOn()
        
    def setPin1Off(self):
        self.set.setPin1Off()
        
    def setPin1On(self):
        self.set.setPin1On()
        
    def setPin2Off(self):
        self.set.setPin1Off()
        
    def setPin2On(self):
        self.set.setPin1On()
        
    def setPin11Off(self):
        self.set.setPin1Off()
        
    def setPin11On(self):
        self.set.setPin1On()
        
    def setPin12Off(self):
        self.set.setPin1Off()
        
    def setPin12On(self):
        self.set.setPin1On()
    
        
        
        
        
        
              

