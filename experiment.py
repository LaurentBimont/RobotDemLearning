from fingertracking import FingerTracker 
import numpy as np
import matplotlib.pyplot as plt



class Validation():
    def __init__(self):
        self.list_eef  = []
        self.list_demo = []
        self.list_ref = []

        self.list_drefe = []
        self.list_dtge = []
        self.list_

    def add_data(self, ref_point, demonstration_point, eef_cart_coordinate):
        self.list_ref.append(ref_point)
        self.list_demo.append(demonstration_point)
        self.list_eef.append( eef_cart_coordinate)

        drefe = (ref_point - eef_cart_coordinate)
        drefe = np.sqrt(derefe.dot(derefe))
        self.list_drefe.append(drefe) 
        
        dtge = (ref_point - demonstration_point)
        dtge = np.sqrt(dtge.dot(dtge))
        self.list_dtge.append(dtge) 

    def show(self):
        pass
