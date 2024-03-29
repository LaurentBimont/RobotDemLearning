import cv2
import numpy as np
from camera import RealCamera
import time
import matplotlib.pyplot as plt
import divers as div

class FingerTracker(object):
    def __init__(self, camera=None):
        super(FingerTracker, self).__init__()
        ## A remettre pour des tests locaux
        # self.cam = RealCamera()
        # self.cam.start_pipe()
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
            try:
                for i in range(len(contour_list)):

                    cnt = contour_list[i]
                    area_cnt = cv2.contourArea(cnt)
                    if area_cnt > max_area:
                        max_area = area_cnt
                        second_i = i
                second_max = contour_list[second_i]
                if (len(second_max) > 5):
                    return first_max, second_max
                else:
                    return first_max, None
            except:
                return first_max, None
        else:
            return None, None

    def centroid(self, max_contour):
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None, None

    def detect_green(self, camera, hist=None):
        self.x_ref, self.y_ref= 50, 50
        self.x_tcp, self.y_tcp = 50, 50
        self.t0 = time.time()
        self.min_over_time = np.inf
        self.list = [] 
        while (time.time() - self.t0) <  5 or len(self.list) ==0:
            print(time.time() - self.t0)
            first_cont, second_cont = None, None

            # Take each frame
            _, frame = camera.get_frame()

            # Convert BGR to HSV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            plt.subplot(2, 2, 1)
            plt.imshow(frame)
            plt.subplot(2, 2, 2)
            plt.imshow(hsv)
            # # define range for green color in HSV
            lower_green = np.array([25, 150, 100])
            upper_green = np.array([40, 255, 250])

            # define range for green color in HSV
            lower_skin = np.array([80, 25, 100], dtype="uint8")
            upper_skin = np.array([140, 130, 255], dtype="uint8")

            # # Threshold the HSV image to get only green colors
            # mask = cv2.inRange(hsv, lower_green, upper_green)
            # Threshold the HSV image to get only skin colors
            if hist is None:
                mask = cv2.inRange(hsv, lower_green, upper_green)
                plt.subplot(2, 2, 3)
                plt.imshow(mask)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.erode(mask, kernel, iterations=1)
                res = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
                plt.subplot(2, 2, 4)
                plt.imshow(mask)

            else:
                res = self.histMasking(frame, hist)
            # Bitwise-AND mask and original image
            gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
            cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            first_cont, second_cont = self.max_contour(cont)

            if first_cont is not None and second_cont is not None:
                cx1, cy1 = self.centroid(first_cont)
                if cx1!= None: 
                    cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)
                 
                    cx2, cy2 = self.centroid(second_cont)
                    if cx2 !=None:
                        cv2.circle(frame, (cx2, cy2), 5, [0, 255, 0], -1)

                        self.x_tcp, self.y_tcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                        self.min_over_time = div.get_ecartement([cx1, cy1, cx2, cy2])
                        self.list = [cx1, cy1, cx2, cy2]

                        cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)

            else:
                cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)

            cv2.circle(frame, (self.x_ref, self.y_ref), 5, [0, 0, 255], -1)
            cv2.imshow('frame', frame)
            # cv2.imshow('res', res)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                return [self.x_tcp, self.y_tcp], self.list

        plt.show()
        cv2.destroyAllWindows()
        return [self.x_tcp, self.y_tcp], self.list

    def detect_red(self, camera, hist=None):
        self.x_ref, self.y_ref= 50, 50
        t0 = time.time()
        self.min_over_time = np.inf
        self.list_ref = None
        while (time.time() - t0) < 10 :
            print(time.time() - t0)
            first_cont, second_cont = None, None

            # Take each frame
            _, frame = camera.get_frame()

            # Convert BGR to HSV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            plt.subplot(2, 2, 1)
            plt.imshow(frame)
            plt.subplot(2, 2, 2)
            plt.imshow(hsv)
            # # define range for red color in HSV
            lower_red= cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
            upper_red= cv2.inRange(hsv, np.array([115,100,100]), np.array([180,255,255]))

            # # Threshold the HSV image to get only green colors
            # mask = cv2.inRange(hsv, lower_green, upper_green)
            # Threshold the HSV image to get only skin colors
            if hist is None:
                mask = lower_red 
                plt.subplot(2, 2, 3)
                plt.imshow(mask)
                kernel = np.ones((3, 3), np.uint8)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                mask = cv2.erode(mask, kernel, iterations=3)
                res = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

            else:
                res = self.histMasking(frame, hist)
            # Bitwise-AND mask and original image
            gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
            cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            first_cont, _ = self.max_contour(cont)
            
            frame = hsv.copy() 
            if first_cont is not None and (not isinstance(first_cont,tuple)) :
                cx1, cy1 = self.centroid(first_cont)

                
                print((cx1,cy1))

                if cx1!=None:
                    cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)

                    self.x_ref, self.y_ref= cx1, cy1

                    self.list_ref = [cx1, cy1]
                    cv2.circle(frame, (self.x_ref, self.y_ref), 5, [0, 0, 255], -1)
                    cv2.put
            else:
                cv2.circle(frame, (self.x_ref, self.y_ref), 5, [0, 0, 255], -1)

            cv2.imshow('frame', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        plt.show()
        cv2.destroyAllWindows()
        return [self.x_ref, self.y_ref], self.list_ref

    def detect_blue(self, camera, hist=None):
        self.x_ref, self.y_ref= 50, 50
        t0 = time.time()
        self.min_over_time = np.inf
        self.list_ref = [] 
        while (time.time() - t0) < 10 and (self.list_ref == [] or len(self.list_ref)!=1) :
            print(time.time() - t0)
            first_cont, second_cont = None, None

            # Take each frame
            _, frame = camera.get_frame()

            # Convert BGR to HSV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            plt.subplot(2, 2, 1)
            plt.imshow(frame_bgr)
            plt.subplot(2, 2, 2)
            plt.imshow(hsv)
            # # define range for red color in HSV
            # blue= cv2.inRange(hsv, np.array([210,100,100]), np.array([255,255,255]))
            # Blue color
            low_blue = np.array([10, 100, 0])
            high_blue = np.array([35, 255, 255])
            blue_mask = cv2.inRange(hsv, low_blue, high_blue)
            # # Threshold the HSV image to get only green colors
            # mask = cv2.inRange(hsv, lower_green, upper_green)
            # Threshold the HSV image to get only skin colors
            if hist is None:
                mask = blue_mask 
                plt.subplot(2, 2, 3)
                plt.imshow(mask)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                mask = cv2.erode(mask, kernel, iterations=3)
                res = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
                plt.subplot(2, 2, 4)
                plt.imshow(mask)
            else:
                res = self.histMasking(frame, hist)
            # Bitwise-AND mask and original image
            gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
            cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            conts = self.max_contour(cont)
            
            self.list_ref = []
            for contour in cont : 
                if contour is not None :
                    cx1, cy1 = self.centroid(contour)

                    print((cx1,cy1))

                    if cx1!=None:
                        cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)

                        self.x_ref, self.y_ref= cx1, cy1

                        self.list_ref.append((cx1, cy1))
                        cv2.circle(frame, (self.x_ref, self.y_ref), 5, [0, 0, 255], -1)
                         
                        P = camera.transform_3D(cx1,cy1)
                        d = np.sqrt(np.dot(P,P) )

                        cv2.putText(frame,"{:1.1f} | {:1.1f} | {:1.1f} \n {:1.1f}".format(*P,d),(cx1-5,cy1-5),cv2.FONT_HERSHEY_PLAIN,0.5,(255,255,255))
                    else:
                        cv2.circle(frame, (self.x_ref, self.y_ref), 5, [0, 0, 255], -1)

            cv2.imshow('frame', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        plt.show()
        cv2.destroyAllWindows()
        return self.list_ref



    def get_frame_without_hand(self, camera):
        self.depth_without_hand, self.frame_without_hand = camera.get_frame()
        return self.depth_without_hand, self.frame_without_hand

    def main(self, camera, viz=True):
        # self.cam.start_pipe()
        depth_without_hand, frame_without_hand = self.get_frame_without_hand(camera)
        tcp, fingers = self.detect_green(camera, hist=None)
        e, angle = div.get_ecartement(fingers), div.get_angle(fingers)
        x, y = tcp[0], tcp[1]

        if viz:
            print(1)
            plt.imshow(frame_without_hand)
            plt.scatter(fingers[0], fingers[1], color='g')
            plt.scatter(fingers[2], fingers[3], color='g')
            plt.scatter(tcp[0], tcp[1], color='r')
            origin = tcp
            vectors = np.array([np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)])
            plt.quiver(*origin, vectors[0], vectors[1], scale=13)
            plt.show()
        # x, y, angle, e = 327, 252, -90, 40
        print(x, y, angle, e, 2*e)
        # self.cam.stop_pipe()
        return x, y, angle, 0.8*e, 1.2*e, depth_without_hand, frame_without_hand

if __name__=="__main__":
    FT = FingerTracker()
    # print(FT.main()[:5])
    # hist = FT.createHistogram()
    # input('Type whatever when you are ready to teach me !')

    print('Show me the way !!')

    # tcp, fingers = FT.detect_green()        # In Image Frame
    #
    # np.save('depth_fingertrack.npy', FT.depth_without_hand)
    # np.save('coordfingers.npy', np.array([fingers]))
    # np.save('coordtcp.npy', np.array(tcp))

    # tcp = np.load('coordtcp.npy')
    # fingers = np.load('coordfingers.npy')[0]
    # depth_without_hand = np.load('depth_fingertrack.npy')

    # Get in Camera Frame
    # TCP_cam = FT.cam.transform_3D(tcp[0], tcp[1], depth_without_hand)
    # P1_cam = FT.cam.transform_3D(fingers[0], fingers[1], depth_without_hand)
    # P2_cam = FT.cam.transform_3D(fingers[2], fingers[3], depth_without_hand)
    camera = RealCamera()  # Vraiment utile ?

    camera.start_pipe()
    time.sleep(1)
    camera_param = [camera.intr.fx, camera.intr.fy, camera.intr.ppx, camera.intr.ppy, camera.depth_scale]
   
    # list_points = FT.detect_blue(camera) 
    FT.detect_blue(camera) 
    # test = list_points[0]


    camera.stop_pipe()



