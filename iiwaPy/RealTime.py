# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:12:34 2018

@author: Mohammad SAFEEA
"""
#from GeneralPurpose import getDoubleFromString
import time

class RealTime:
    
    def __init__(self,mysoc):
        self.mysoc=mysoc

    def send(self,data):
        data=data+b'\n'
        self.mysoc.send(data)
        self.mysoc.receive()
        
    def realTime_stopImpedanceJoints(self):
        theCommand=b'stopDirectServoJoints'
        self.send(theCommand)
        time.sleep(0.3)
        
    def realTime_stopDirectServoJoints(self):
        theCommand=b'stopDirectServoJoints'
        self.send(theCommand)
        time.sleep(0.3)
        
    def realTime_startDirectServoJoints(self):     
        theCommand=b'startDirectServoJoints'
        self.send(theCommand)
        time.sleep(0.3)
        
    def realTime_startImpedanceJoints(self,weightOfTool,cOMx,cOMy,cOMz,cStiness,rStifness,nStifness):
        theCommand=b'startSmartImpedneceJoints'
        theCommand=theCommand+b'_'+bytes(weightOfTool)
        theCommand=theCommand+b'_'+bytes(cOMx)
        theCommand=theCommand+b'_'+bytes(cOMy)
        theCommand=theCommand+b'_'+bytes(cOMz)
        theCommand=theCommand+b'_'+bytes(cStiness)
        theCommand=theCommand+b'_'+bytes(rStifness)
        theCommand=theCommand+b'_'+bytes(nStifness)+b'_'
        self.send(theCommand)
        time.sleep(0.3)
