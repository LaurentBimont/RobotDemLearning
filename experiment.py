from fingertracking import FingerTracker 
import numpy as np
import matplotlib.pyplot as plt
from camera import RealCamera


class Validation():
    def __init__(self):
        self.list_decision  = []
        self.list_demo = []
        self.list_ref = []

        self.list_dref_demo = []
        self.list_dref_decision = []

    def compute_distance(self, ref_point, demonstration_point, decision_point):
        self.list_ref.append(ref_point)
        self.list_demo.append(demonstration_point)
        self.list_decision.append( decision_point)

        #distance projected on the base plane 
        dref_decision = (ref_point[:2] - decision_point[:2])
        dref_decision = np.sqrt(np.dot(dref_decision, dref_decision))
        self.list_dref_decision.append(dref_decision) 

        #distance projected on the base plane 
        dref_demo= (ref_point[:2] - demonstration_point[:2])
        dref_demo= np.sqrt(np.dot(dref_demo,dref_demo))
        self.list_dref_demo.append(dref_demo) 


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
