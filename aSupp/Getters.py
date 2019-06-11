# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:42:39 2018

@author: Mohammad SAFEEA
"""

from iiwaaControler.GeneralPurpose import getDoubleFromString


class Getters:
    def __init__(self, mysoc):
        self.mysoc = mysoc

    def send(self, data, size):
        data = data + b'\n'
        self.mysoc.send(data)
        message = self.mysoc.receive()
        # print(message)
        # sys.stdout.flush()
        return getDoubleFromString(message, size)

    def sendShort(self, data):
        data = data + b'\n'
        self.mysoc.send(data)
        message = self.mysoc.receive()
        return float(message)

# getters

    def getEEFPos(self):
        theCommand = b'Eef_pos'
        eefPos = self.send(theCommand, 6)
        return eefPos

    def getEEF_Force(self):
        theCommand = b'Eef_force'
        force = self.send(theCommand, 3)
        return force

    def getEEFCartesianPosition(self):
        theCommand = b'Eef_pos'
        eefPos = self.send(theCommand, 3)
        return eefPos

    def getEEF_Moment(self):
        theCommand = b'Eef_moment'
        moment = self.send(theCommand, 3)
        return moment

    def getJointsPos(self):
        theCommand = b'getJointsPositions'
        jointsPos = self.send(theCommand, 7)
        return jointsPos

    def getJointsExternalTorques(self):
        theCommand = b'Torques_ext_J'
        taw = self.send(theCommand, 7)
        return taw

    def getJointsMeasuredTorques(self):
        theCommand = b'Torques_m_J'
        taw = self.send(theCommand, 7)
        return taw

    def getMeasuredTorqueAtJoint(self, x):
        theCommand = b'Torques_m_J'
        taw = self.send(theCommand, 7)
        return taw[
            x - 1]  # array index starts from zero, joint index start from one

    def getEEFCartesianOrientation(self):
        theCommand = b'Eef_pos'
        eefPos = self.send(theCommand, 6)
        return eefPos[3:6]


# get pin states

    def getPin3State(self):
        theCommand = b'getPin3'
        state = self.sendShort(theCommand)
        return state

    def getPin4State(self):
        theCommand = b'getPin4'
        state = self.sendShort(theCommand)
        return state

    def getPin10State(self):
        theCommand = b'getPin10'
        state = self.sendShort(theCommand)
        return state

    def getPin13State(self):
        theCommand = b'getPin13'
        state = self.sendShort(theCommand)
        return state

    def getPin16State(self):
        theCommand = b'getPin16'
        state = self.sendShort(theCommand)
        return state
