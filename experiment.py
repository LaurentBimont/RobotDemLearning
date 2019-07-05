from fingertracking import FingerTracker 
import numpy as np
import matplotlib.pyplot as plt
from camera import RealCamera


class Validation():
    def __init__(self):
        self.list_eef  = []
        self.list_demo = []
        self.list_ref = []

        self.list_drefe = []
        self.list_dtge = []

    def compute_distance(self, ref_point, demonstration_point, eef_cart_coordinate):
        self.list_ref.append(ref_point)
        self.list_demo.append(demonstration_point)
        self.list_eef.append( eef_cart_coordinate)

        #distance projected on the base plane 
        drefe = (ref_point[:2] - eef_cart_coordinate[:2])
        drefe = np.sqrt(np.dot(drefe,drefe))
        self.list_drefe.append(drefe) 

        #distance projected on the base plane 
        dtge = (ref_point[:2] - demonstration_point[:2])
        dtge = np.sqrt(np.dot(dtge,dtge))
        self.list_dtge.append(dtge) 

        dteeftg =  (demonstration_point[:2] - eef_cart_coordinate[:2])
        dteeftg= np.sqrt(np.dot(dteeftg,dteeftg))

        return drefe, dtge, dteeftg  

    def show(self):
        pass
if __name__=="__main__":
    camera = RealCamera()
    camera.start_pipe()
    FT = FingerTracker(camera)
    ref_points = FT.detect_blue(camera)

    val = Validation() 

    for point in ref_points :
        FT.x_ref, FT.y_ref = point

        demo_point = FT.detect_green(camera)[0]


        ref_point3D = camera.transform_3D(*point)
        demo_point3D = camera.transform_3D(*demo_point) 
        simu_eef_point3D = (demo_point3D[0] +5, demo_point3D[1]+5,demo_point3D[2]-2)

        print(val.compute_distance(ref_point3D, demo_point3D, simu_eef_point3D))
    camera.stop_pipe()
