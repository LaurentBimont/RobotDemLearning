# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:44:35 2018

@author: Mohammad SAFEEA
"""
import math
from io import BytesIO
from iiwaaControler.Senders import Senders
from iiwaaControler.Getters import Getters
from iiwaaControler.check import check_size, check_scalar, check_non_zero
import numpy as np


def checkAcknowledgment(msg):
    ak="done"
    nak="nak"
    print(msg)
    if msg=="":
        return False
    if (msg==nak):
        return False
    if msg==ak:
        return True
    else:
        return False
class PTP:
    
    def __init__(self,mysoc):
        self.mysoc=mysoc
        self.sender=Senders(mysoc)
        self.getter=Getters(mysoc)

    def send(self,data):
        data=data+b"\n"
        
        self.mysoc.send(data)
        msg = self.mysoc.receive()
        return msg

    def createCommand(self,name, data ):
        command_list = [name]+[str(data)]+ [""]
        command = "_".join(command_list).encode("ascii")
        return command

    def blockUntilAcknowledgment(self):
        while(True):
            msg = self.mysoc.receive()
            if (checkAcknowledgment(msg)):
                break

    ## arcc motions
    def movePTPArc_AC(self,theta,c,k,vel):
        check_size(3,"Center of circle", c)
        check_size(3,"Orientation vector", k)
        check_scalar("Angle",theta )
        check_scalar("Relative velocity", vel)
        
        pos=self.getter.getEEFPos()
        x=math.pow(c[0]-pos[0],2)+math.pow(c[1]-pos[1],2)+math.pow(c[2]-pos[2],2)
        r=math.pow(x,0.5) 

        check_non_zero("Radius", r)
        check_non_zero("Angle",theta)

        #calculate unit vector
        x=math.pow(k[0],2)+math.pow(k[1],2)+math.pow(k[2],2)
        normK=math.pow(x,0.5)
        check_non_zero("Norm with direction of vector k", normK)

        k[0]=k[0]/normK
        k[1]=k[1]/normK
        k[2]=k[2]/normK
        s=[c[0]-pos[0],c[1]-pos[1],c[2]-pos[2]]
        s[0]=-s[0]/r
        s[1]=-s[1]/r
        s[2]=-s[2]/r
        n=[(k[1]*s[2]-k[2]*s[1]),(k[2]*s[0]-k[0]*s[2]),(k[0]*s[1]-k[1]*s[0])]
        c1 = self.rotTheThing(theta/2,r,s,n,c)
        c2 = self.rotTheThing(theta,r,s,n,c)
        self.movePTPCirc1OrintationInter(c1,c2, vel)
         
    def rotTheThing(self,theta,r,s,n,c):
        c1=[0,0,0]
        cos=math.cos(theta)
        sin=math.sin(theta)
        c1[0]=r*cos*s[0]+r*sin*n[0]+c[0]
        c1[1]=r*cos*s[1]+r*sin*n[1]+c[1]
        c1[2]=r*cos*s[2]+r*sin*n[2]+c[2]
        return c1
        
    def movePTPArcXY_AC(self,theta,c,vel):
        check_size(2,"Center of circle", c)
        check_scalar("Angle",theta )
        check_scalar("Relative velocity", vel)

        k=[0,0,1]
        pos=self.getter.getEEFPos()
        c1=[c,pos[2]]
        self.movePTPArc_AC(theta,c1,k,vel)
        
    def movePTPArcXZ_AC(self,theta,c,vel):

        check_size(2,"Center of circle", c)
        check_scalar("Angle",theta )
        check_scalar("Relative velocity", vel)
        
        k=[0,1,0]
        pos=self.getter.getEEFPos()
        c1=[c[0],pos[1],c[1]]
        self.movePTPArc_AC(theta,c1,k,vel)
        
    def movePTPArcYZ_AC(self,theta,c,vel):

        check_size(2,"Center of circle", c)
        check_scalar("Angle",theta )
        check_scalar("Relative velocity", vel)
        
        k=[1,0,0]
        pos=self.getter.getEEFPos()
        c1=[pos[0],c[1],c[2]]
        self.movePTPArc_AC(theta,c1,k,vel)
        
    def movePTPCirc1OrintationInter(self, f1,f2, relVel):
        check_size(6,"First frame [x,y,z,alpha,beta,gamma]", f1)
        check_size(6,"Second frame [x,y,z,alpha,beta,gamma]", f2)
        check_scalar("Relative velocity", relVel)

        command=self.createCommand('jRelVel',relVel) 
        self.send(command)

        self.sender.sendCirc1FramePos(f1)
        self.sender.sendCirc2FramePos(f2)

        theCommand=b'doPTPinCSCircle1_'
        self.send(theCommand)
        self.blockUntilAcknowledgment()
    def movePTPLineEEF(self,pos,vel):
        check_size(6,"Position ", pos)
        check_scalar("Velocity", vel)

        command = self.createCommand("jRelVel",vel)
        
        #
        self.send(command)
        self.sender.sendEEfPositions(pos)

        theCommand=b'doPTPinCS_'
        self.send(theCommand)
        self.blockUntilAcknowledgment() 
            
    def movePTPLineEEFRelEEF(self,pos,vel):
        check_size(3,"Position ", pos)
        check_scalar("Velocity", vel)

        command = self.createCommand("jRelVel",vel)

        self.send(command)
        newPos=[0,0,0,0,0,0]
        newPos[0]=pos[0]
        newPos[1]=pos[1]
        newPos[2]=pos[2]

        self.sender.sendEEfPositions(newPos)
        theCommand=b'doPTPinCSRelEEF_'
        self.send(theCommand)
        self.blockUntilAcknowledgment()

    def movePTPLineEEFRelBase(self,pos,vel):
        check_size(3,"Position ", pos)
        check_scalar("Velocity", vel)

        command = self.createCommand("jRelVel",vel)
        self.send(command)

        newPos=[0,0,0,0,0,0,0]
        newPos[0]=pos[0]
        newPos[1]=pos[1]
        newPos[2]=pos[2]
        self.sender.sendEEfPositions(newPos)

        theCommand=b'doPTPinCSRelBase'
        self.send(theCommand)
        self.blockUntilAcknowledgment()
# space
    def movePTPJointSpace(self,jpos,relVel):
        check_size(7,"Joints ",jpos)
        check_scalar("Relative Velocity", relVel)

        command = self.createCommand("jRelVel",relVel)
        self.send(command)

        self.sender.sendJointsPositions(jpos)

        theCommand=b'doPTPinJS'
        self.send(theCommand)
        self.blockUntilAcknowledgment()
    def movePTPHomeJointSpace(self,relVel):
        check_scalar("Relative Velocity", relVel)

        command = self.createCommand("jRelVel",relVel)
        self.send(command)

        jpos=[0,0,0,0,0,0,0]
        self.sender.sendJointsPositions(jpos)

        theCommand=b'doPTPinJS_'
        self.send(theCommand)
        
        self.blockUntilAcknowledgment()
        
    def movePTPTransportPositionJointSpace(self,relVel):
        check_scalar("Relative Velocity", relVel)
        jpos=[0,0,0,0,0,0,0]
        jpos[3]=25*math.pi/180
        jpos[5]=90*math.pi/180
        self.movePTPJointSpace(jpos,relVel)
