# -*- coding: utf-8 -*- % % 
"""
Created on Tue Mar 27 17:36:18 2018

@author: Mohammad SAFEEA
"""

from io import BytesIO
import math
import numpy as np
from iiwaaControler.GeneralPurpose import getDoubleFromString
from iiwaaControler.check import check_size


class Senders:
    
    def __init__(self,mysoc):
        self.mysoc=mysoc

    def send(self,data):
        data=data+b'\n'
        self.mysoc.send(data)
        message=self.mysoc.receive()
        return message

    def createCommand(self,name, data_list, float_precision=4):
        command_list = [name]+list(map(str, np.round(data_list,float_precision))) + [""]
        command = "_".join(command_list).encode("ascii")
        return command
    
    def sendEEfPositions(self,x):
        check_size(6,"EEF positions",x)
        command = self.createCommand("cArtixanPosition",x)
        self.send(command)


    def sendJointsPositions(self,x):
        check_size(7,"Joint positions",x)
        command = self.createCommand("jp",x)
        self.send(command)


    def sendJointsPositionsGetMTorque(self,x):
        check_size(7,"Joint positions",x)
        command = self.createCommand("jpMT",x)
        self.send(command)

        return getDoubleFromString(self.send(command),7)


    def sendJointsPositionsGetExTorque(self,x):
        check_size(7,"Joint positions",x)
        command = self.createCommand("jpExT",x)
        self.send(command)

        return getDoubleFromString(self.send(command),7)


    def sendJointsPositionsGetActualJpos(self,x):
        check_size(7,"Joint positions",x)
        command = self.createCommand("jpJP",x)
        self.send(command)

        return getDoubleFromString(self.send(command),7)


    def sendCirc1FramePos(self,fpos):
        check_size(6,"Frame coordinate [x,y,z,alpha,beta,gamma]",x)
        command = self.createCommand("cArtixanPositionCirc1",x)
        self.send(command)


    def sendCirc2FramePos(self,fpos):
        check_size(6,"Frame coordinate [x,y,z,alpha,beta,gamma]",x)
        command = self.createCommand("cArtixanPositionCirc2",x)
        self.send(command)
