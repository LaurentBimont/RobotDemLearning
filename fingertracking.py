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

    def createHistogram(self):
        t0 = time.time()
        width = 50
        while (time.time() - t0)<5:
            _, frame = self.cam.get_frame()
            rows, cols, _ = frame.shape
            y0, x0 = int(0.5 * rows), int(0.2 * cols)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (x0, y0), (x0+width, y0+width), (255,0,0), 2)
            cv2.imshow('frame', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()

        rows, cols, _ = frame.shape
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([width, width], dtype=hsvFrame.dtype)
        y0, x0 = int(0.5 * rows), int(0.2 * cols)
        roi = hsvFrame[y0:y0 + width, x0:x0 + width]
        hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    def histMasking(self, frame, hist):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        # thresh = cv2.dilate(thresh, kernel, iterations=5)
        # Erode : Only if all pixel under mask are 1
        # Dilate : If at least one pixel under mask is 1
        # Opening : Erosion then Dilatation
        # Closing : Dilatation then Erosion

        thresh = cv2.erode(thresh, kernel, iterations=5)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(frame, thresh)

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
                if (len(second_max) > 50):
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
            return None

    def detect_green(self, hist=None):
        self.x_tcp, self.y_tcp = 50, 50
        self.t0 = time.time()
        self.min_over_time = np.inf
        self.list = None
        while (time.time() - self.t0) < 5 and self.list is None:
            print(type(self.list))
            first_cont, second_cont = None, None

            # Take each frame
            self.cam.get_frame()
            frame = self.cam.color_image
            _, frame = self.cam.get_frame()

            # Convert BGR to HSV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            plt.subplot(2, 2, 1)
            plt.imshow(frame)
            plt.subplot(2, 2, 2)
            plt.imshow(hsv)
            # # define range for green color in HSV
            # lower_green = np.array([60, 40, 40])
            # upper_green = np.array([90, 250, 250])

            # define range for green color in HSV
            lower_skin = np.array([80, 25, 100], dtype="uint8")
            upper_skin = np.array([140, 130, 255], dtype="uint8")

            # # Threshold the HSV image to get only green colors
            # mask = cv2.inRange(hsv, lower_green, upper_green)
            # Threshold the HSV image to get only skin colors
            if hist is None:
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                plt.subplot(2, 2, 3)
                plt.imshow(mask)
                kernel = np.ones((4, 4), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                mask = cv2.erode(mask, kernel, iterations=3)
                res = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
                plt.subplot(2, 2, 4)
                plt.imshow(mask)
                plt.show()
            else:
                res = self.histMasking(frame, hist)
            # Bitwise-AND mask and original image

            # plt.subplot(1, 2, 1)
            # plt.imshow(frame)
            # plt.subplot(1, 2, 2)
            # plt.imshow(res)
            # plt.show()

            gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
            # plt.imshow(thresh)
            # plt.show()
            cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            first_cont, second_cont = self.max_contour(cont)
            if first_cont is not None and second_cont is not None:
                cx1, cy1 = self.centroid(first_cont)
                cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)

                cx2, cy2 = self.centroid(second_cont)
                cv2.circle(frame, (cx2, cy2), 5, [0, 255, 0], -1)

                print(self.get_ecartement([cx1, cy1, cx2, cy2]), self.min_over_time)

                # if self.get_ecartement([cx1, cy1, cx2, cy2]) < self.min_over_time:
                #     self.x_tcp, self.y_tcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                #     self.min_over_time = self.get_ecartement([cx1, cy1, cx2, cy2])
                #     print('Voici mon min : ', self.min_over_time)
                #     self.list = [cx1, cy1, cx2, cy2]

                self.x_tcp, self.y_tcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                self.min_over_time = self.get_ecartement([cx1, cy1, cx2, cy2])
                print('Voici mon min : ', self.min_over_time)
                self.list = [cx1, cy1, cx2, cy2]

                cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)
            else:
                cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)

            cv2.imshow('frame', frame)
            cv2.imshow('res', res)

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
        return self.depth_without_hand, self.frame_without_hand

    def get_angle(self, fingers):
        u1, v1, u2, v2 = fingers[0], fingers[1], fingers[2], fingers[3]
        angle = 180/np.pi * div.py_ang(np.array([u1-u2, v1-v2]), np.array([1, 0]))
        return(angle - 90)

    def main(self, viz=True):
        depth_without_hand, frame_without_hand = self.get_frame_without_hand()
        # tcp, fingers = self.detect_green(hist=None)
        # e, angle = self.get_ecartement(fingers), self.get_angle(fingers)
        # x, y = tcp[0], tcp[1]

        # if viz:
        #     print(1)
        #     plt.imshow(frame_without_hand)
        #     plt.scatter(fingers[0], fingers[1], color='g')
        #     plt.scatter(fingers[2], fingers[3], color='g')
        #     plt.scatter(tcp[0], tcp[1], color='r')
        #     origin = tcp
        #     vectors = np.array([np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)])
        #     plt.quiver(*origin, vectors[0], vectors[1], scale=13)
        #     plt.show()
        x, y, angle, e = 217, 184, -90, 20
        self.cam.stop_pipe()
        return x, y, angle, e, 2*e, depth_without_hand, frame_without_hand

if __name__=="__main__":
    FT = FingerTracker()
    print(FT.main())
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
    #
    # TCP_cam = FT.cam.transform_3D(tcp[0], tcp[1], depth_without_hand)
    # P1_cam = FT.cam.transform_3D(fingers[0], fingers[1], depth_without_hand)
    # P2_cam = FT.cam.transform_3D(fingers[2], fingers[3], depth_without_hand)


    FT.cam.stop_pipe()


