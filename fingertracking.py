import cv2
import numpy as np
from camera import RealCamera
import time
import matplotlib.pyplot as plt
import divers as div

class FingerTracker(object):
    def __init__(self):
        super(FingerTracker, self).__init__()
        self.cam = RealCamera()
        self.cam.start_pipe()
        self.t0 = time.time()
        self.min_over_time = np.inf
        self.x_tcp, self.y_tcp = None, None

    def max_contour(self, contour_list):
        first_i, second_i = 0, 0
        max_area = 0
        if len(contour_list) != 0:
            for i in range(len(contour_list)):
                cnt = contour_list[i]
                area_cnt = cv2.contourArea(cnt)
                if area_cnt > max_area:
                    max_area = area_cnt
                    first_i = i
            first_max = contour_list[first_i]
            contour_list = np.delete(contour_list, first_i)

            max_area = 0
            for i in range(len(contour_list)):
                cnt = contour_list[i]
                area_cnt = cv2.contourArea(cnt)
                if area_cnt > max_area:
                    max_area = area_cnt
                    second_i = i
            second_max = contour_list[second_i]
            if (len(second_max) > 50):
                return first_max, second_max
            else:
                return None, None
        else:
            return None

    def centroid(self, max_contour):
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None

    def detect_green(self):
        self.x_tcp, self.y_tcp = 50, 50
        self.t0 = time.time()
        self.min_over_time = np.inf
        print('Pas Normal')
        while (time.time() - self.t0) < 10:

            first_cont, second_cont = None, None

            # Take each frame
            self.cam.get_frame()
            frame = self.cam.color_image
            _, frame = self.cam.get_frame()
            # Convert BGR to HSV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # define range for green color in HSV
            lower_green = np.array([60, 40, 40])
            upper_green = np.array([90, 250, 250])

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)
            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame, frame, mask=mask)

            gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
            cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            first_cont, second_cont = self.max_contour(cont)
            if first_cont is not None and second_cont is not None:
                cx1, cy1 = self.centroid(first_cont)
                cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)

                cx2, cy2 = self.centroid(second_cont)
                cv2.circle(frame, (cx2, cy2), 5, [0, 255, 0], -1)

                print(FT.get_ecartement(cx1, cy1, cx2, cy2), self.min_over_time)

                if FT.get_ecartement(cx1, cy1, cx2, cy2) < self.min_over_time:
                    self.x_tcp, self.y_tcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                    self.min_over_time = FT.get_ecartement(cx1, cy1, cx2, cy2)
                    print('Voici mon min : ', self.min_over_time)
                    self.list = [cx1, cy1, cx2, cy2]

                cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)
            else:
                cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)

            cv2.imshow('frame', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        return [self.x_tcp, self.y_tcp], self.list

    def get_ecartement(self, fingers):
        # return np.linalg.norm(self.cam.transform_3D(u1, v1)-self.cam.transform_3D(u2, v2))
        u1, v1, u2, v2 = fingers[0], fingers[1], fingers[2], fingers[3]
        ##### Temporaire : renvoie juste la difference en pixel #######"
        return np.linalg.norm(np.array([[u1-u2], [v1-v2]]))

    def get_frame_without_hand(self):
        self.depth_without_hand, self.frame_without_hand = self.cam.get_frame()

    def get_angle(self, fingers):
        u1, v1, u2, v2 = fingers[0], fingers[1], fingers[2], fingers[3]
        angle = 180/np.pi * div.py_ang(np.array([u1-u2, v1-v2]), np.array([1, 0]))
        return(angle - 90)

if __name__=="__main__":
    FT = FingerTracker()

    # input('Type whatever when you are ready to teach me !')

    FT.get_frame_without_hand()

    print('Show me the way !!')

    # tcp, fingers = FT.detect_green()        # In Image Frame
    #
    # np.save('depth_fingertrack.npy', FT.depth_without_hand)
    # np.save('coordfingers.npy', np.array([fingers]))
    # np.save('coordtcp.npy', np.array(tcp))

    tcp = np.load('coordtcp.npy')
    fingers = np.load('coordfingers.npy')[0]
    depth_without_hand = np.load('depth_fingertrack.npy')

    # Get in Camera Frame


    TCP_cam = FT.cam.transform_3D(tcp[0], tcp[1], depth_without_hand)
    P1_cam = FT.cam.transform_3D(fingers[0], fingers[1], depth_without_hand)
    P2_cam = FT.cam.transform_3D(fingers[2], fingers[3], depth_without_hand)

    print('L\'ecartement est {} pixels et l\'angle est {} degrés'.format(FT.get_ecartement(fingers), FT.get_angle(fingers)))

    e, angle = FT.get_ecartement(fingers), FT.get_angle(fingers)

    nuage = FT.cam.transform_image_to_3D(depth_without_hand)

    plt.imshow(nuage[:, :, 2])
    plt.scatter(fingers[0], fingers[1], color='g')
    plt.scatter(fingers[2], fingers[3], color='g')
    plt.scatter(tcp[0], tcp[1], color='r')
    origin = tcp
    vectors = np.array([np.cos(angle), np.sin(angle)])
    plt.quiver(*origin, vectors[0], vectors[1], scale=13)
    plt.show()
