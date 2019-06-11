# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:45:29 2018

@author: Mohammad SAFEEA
"""

import sys
import numpy as np

def getDoubleFromString(message,size):

    strVals=message.split("_")
    if strVals[-1]=="":
        strVals.pop(-1)
    try:
        doubleVals=np.array(list(map(float,strVals)))
    except ValueError:
        raise ValueError('can not convert the following array to float  array {}'.format(strVals))
            
    return doubleVals
    
def directKinematics(q):
    if(len(q)!=7):
        print('Error in function [directKinematics]')
        print('The size of the joint angles shall be 7')
        return
