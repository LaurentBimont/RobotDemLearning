from iiwaPy.sunrisePy import sunrisePy
import time
import numpy as np
from gripper import RobotiqGripper
import matplotlib.pyplot as plt
import divers as div
import sys, os
import traceback
# TEST TEMPORAIRE POUR CREER LA FONCTION camera-->robot
from camera import RealCamera

class Robot:
    def __init__(self):

        # Define CameraTransormation Matrix
        self.camera = RealCamera()
        self.Rmain_camera = np.load('calib/CalibrationRotation.npy')
        self.Tmain_camera = np.load('calib/CalibrationTranslation.npy')

        # Connection to gripper
        self.grip = RobotiqGripper("/dev/ttyUSB0")
        self.grip.reset()
        self.grip.activate()
        self.grip.closeGripper()

        # Connection to robot
        ip = '172.31.1.148'
        self.iiwa = sunrisePy(ip)
        self.iiwa.setBlueOn()

        time.sleep(2)
        self.iiwa.setBlueOff()
        self.relVel = 0.1
        self.vel = 10
        self.iiwa.attachToolToFlange([-1.5, 1.54, 252.8, 0, 0, 0])
        self.z_min = 12.3       # Z mousse
        self.z_min = -18      # Z table

    def getCart(self):
        return self.iiwa.getEEFCartesianPosition()

    def manual_click(self, depth_img):
        fig = plt.figure()
        connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
        print(depth_img.shape, type(depth_img))
        plt.imshow(depth_img)
        plt.show()
        while True:
            try:
                x = x_click
                y = y_click
                break
            except:
                pass
        return x, y

    def camera2robot(self):
        global x_click, y_click
        # Compute tool orientation from heightmap rotation angle
        position = self.manual_click()

        Rbase_main, Tbase_main = self.RetT_Matrix()

        print('Position ', position)
        print('Rotation', self.R_cam_poign, Rbase_main)
        print('Translation', self.T_cam_poign, Tbase_main)
        position = np.dot(self.R_cam_poign, position) + self.T_cam_poign
        position = np.dot(np.transpose(Rbase_main), position) - Tbase_main
        print('La position est: ', position)

    ##### Geometric Transformation #####
    def get_rotation_translation(self, cart_pos):
        x_tcp, y_tcp, z_tcp, A_tcp, B_tcp, C_tcp = cart_pos
        # With Euler Representation
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(C_tcp), -np.sin(C_tcp)],
                       [0, np.sin(C_tcp), np.cos(C_tcp)]])
        Ry = np.array([[np.cos(B_tcp), 0, np.sin(B_tcp)],
                       [0, 1, 0],
                       [-np.sin(B_tcp), 0, np.cos(B_tcp)]])
        Rz = np.array([[np.cos(A_tcp), -np.sin(A_tcp), 0],
                       [np.sin(A_tcp), np.cos(A_tcp), 0],
                       [0, 0, 1]])
        Rbase_main = Rz.dot(Ry.dot(Rx))
        Tbase_main_repere_base = np.array([x_tcp, y_tcp, z_tcp])
        return Rbase_main, Tbase_main_repere_base

    def fromAtoB(self, PA, Ra2b, Ta2b_basea, changeT=True):
        '''
        Fait les calculs suivants : Pb = Ra2b.T x Pa - Ra2b.T x Ta2b_basea
        :param PA: Coordonné d'un point dans le repère A
        :param Ra2b: Rotation de A vers B
        :param Ta2b_basea: Vecteur de translation OaOb exprimé dans la base A
        :return: PB : Coordonnées du point P dans le repère B
        '''
        if changeT:
            T = np.transpose(Ra2b).dot(np.transpose(Ta2b_basea))
        else:  # Cas ou l'utilisateur rentre directement la bonne translation
            T = Ta2b_basea
        Ta2b_baseb = np.transpose(np.tile(T, (PA.shape[1], 1)))
        PB = np.transpose(Ra2b).dot(PA) - Ta2b_baseb
        return PB

    def camera2main(self, camera, R, T):
        print(np.transpose(np.tile(T, (camera.shape[0], 1))).shape, np.transpose(np.tile(T, (camera.shape[0], 1))))
        return np.transpose(R).dot(np.transpose(camera) - np.transpose(np.tile(T, (camera.shape[0], 1))))

    def pixel2camera(self, depth_img, u, v):
        camera = self.camera.transform_3D(u, v, depth_img)
        return np.transpose(camera)

    def from_camera2robot(self, depth_img, u, v, color=None, camera_param=None, cartpos=None, test_manuel=False):
        '''
        Transform from camera coordinates to robot coordinates
        :param camera: Coordonnées dans le repère caméra (shape : (3, 1) np.array([[x],[y],[z]]))
        :param cartpos: Coordonnées cartésiennes du robot
        :return:
        '''
        if test_manuel:
            if color is None:
                u, v = self.manual_click(depth_img)
            else:
                u, v = self.manual_click(color)
        second_min = div.second_min(depth_img.flatten())
        depth_img[depth_img == 0.] = second_min
        camera = self.camera.transform_3D(u, v, image=depth_img, param=camera_param)
        camera = camera.reshape(3, 1)
        if cartpos is None:
            cartpos = self.getCart()
        Base_main = self.camera2main(np.transpose(camera), self.Rmain_camera, self.Tmain_camera)
        R, T = self.get_rotation_translation(cartpos)
        Base_robot = self.fromAtoB(Base_main, np.transpose(R), -T, changeT=False)
        Base_robot = Base_robot.reshape(3)
        return Base_robot

    def checkTCPJoint(self):
        cartPos = self.iiwa.getEEFCartesianPosition()
        vel = 1
        orientationVel = 0.1
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)
        print('Autour de C')
        cartPos[5] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)
        cartPos[5] -= np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)
        print('Autour de B')
        cartPos[4] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)
        cartPos[4] -= np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)
        print('Autour de A')
        cartPos[3] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)
        cartPos[3] -= np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel, orientationVel=orientationVel)

    def moveto(self, cart):
        #cart[2] = max(cart[2], -18.64)
        ## Only for angle test
        OK='1'
        while OK=='1':
            angle = int(input('Rentrer l\'angle : '))
            goto = [450, -2.13, 329, angle*np.pi/180, 0, np.pi]
            goto = [cart[0], cart[1], cart[2], angle*np.pi/180, 0, np.pi]
            speed = 20
            self.iiwa.movePTPLineEEF(goto, speed, orientationVel=0.1)
            OK = input('Continuer ? : ')

    def home(self):
        speed = 100
        home = [595, 15.9, 353, 1.60, 0, 2.77] # Orientation camera vers l'arrière
        goto = [383, -2.13, 250, -1.63, 0, 2.72] # Orientation camera vers l'avant
        angular_home = [0.19611477, 0.430804, -0.30726947, -1.47291056, 0.10567891, 1.67921869, -1.63443118]
        angular_home = [0.19540027, 0.43081131, -0.30892252, -1.47040016, 0.12090041, 1.62190036, -1.64482995]
        # angular_home = [ 0.22088622,  0.660586  , -0.30721297, -1.63273026,  0.19087189,
        # 1.30512739, -1.66725927] 
        self.iiwa.movePTPJointSpace(angular_home, 0.50)
        # self.iiwa.movePTPLineEEF(goto, speed, orientationVel=0.5)

    def grasp(self, pos, ang, speed=50, rotate=False):
        angle = div.angle2robotangle(ang)*np.pi/180
        pos[2] = max(pos[2]-60, self.z_min)
        pos[2] -= 5

        pos[0] += 30
        pos[1] -= 30.7
        print('L angle dans le repère robot : {} \nAngle dans le repère base{}'.format(angle, ang))
        grasp_above = [pos[0], pos[1], pos[2]+100., angle, 0, np.pi]
        print('Grasping à ', grasp_above)
        self.iiwa.movePTPLineEEF(grasp_above, 8*speed, orientationVel=1)

        self.grip.goTo(75)
        grasp = [pos[0], pos[1], pos[2], angle, 0, np.pi]
        self.iiwa.movePTPLineEEF(grasp, speed, orientationVel=0.1)
        self.grip.closeGripper()
        time.sleep(1)
        grasp_success = self.grip.isObjectDetected()
        print('Objet Détectée', grasp_success)

        self.iiwa.movePTPLineEEF(grasp_above, 8*speed, orientationVel=1)
        self.iiwa.movePTPLineEEF([814, -52.5, 283.6, -1.64, -0, -2.36], 8*speed, orientationVel=1)
        input('Tapez quand vous etes pret a recuperer : ')
        self.grip.goTo(60)
        self.home()
        return grasp_success

def onclick(event):
    global x_click, y_click
    x_click, y_click = int(event.xdata), int(event.ydata)
    print(x_click, y_click)

if __name__=="__main__":
    rob = Robot()
    # rob.home()
    # pos = [4.93012815e+02, -5.54576652e+00, 2.11998124e+02, angle, 0, np.pi]
    print('Pos', rob.iiwa.getEEFPos())
    print('PosCart', rob.iiwa.getEEFCartesianPosition())
    print('PosAngle', rob.iiwa.getEEFCartesianOrientation())
    print('Angle Actuelle : ', rob.iiwa.getEEFPos()[3]*180/np.pi)
    # grasp_above = [, ,  angle, 0, np.pi]
    # angle_commanfe = rob.iiwa.getEEFPos()[3]*180/np.pi
    # for i in range(300):
    #     angle_commanfe = angle_commanfe + 1
    #     speed = 20
    #     print(angle_commanfe)
    #     angle = angle_commanfe * np.pi/180
    #     pos = [494, -5, 211, angle, 0, np.pi]
    #     rob.iiwa.movePTPLineEEF(pos, speed, orientationVel=0.5)
    #     print('Sleeping in progress...')
    #     time.sleep(0.05)
    try:
        # camera = RealCamera()
        # camera.start_pipe()
        # camera_param = [camera.intr.fx, camera.intr.fy, camera.intr.ppx, camera.intr.ppy, camera.depth_scale]
        # camera.get_frame()
        # print(camera.color_image)
        # print(rob.getCart())
        # rob = Robot()
        # rob.home()
        # print('Joint Position', rob.iiwa.getJointsPos())
        # cart = rob.from_camera2robot(camera.depth_image, 241, 290, color=camera.color_image, camera_param=camera_param, test_manuel=True)
        # # rob.moveto([[450, -2.13, 329]])
        # OK = input('Move à la position cartésienne : {}  Est-ce OK ? (oui : 1)'.format(cart))
        # rob.moveto(cart)
        # ang = [-1.5, 0., np.pi]
        # rob.grasp(cart, ang)
        # print('Position Cartésienne atteinte', rob.getCart())
        # rob.camera.stop_pipe()
        rob.iiwa.close()
    except Exception as e:
        exc_info = sys.exc_info()
        print(e)
        # rob.camera.stop_pipe()
        rob.iiwa.close()
