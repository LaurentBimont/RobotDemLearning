from iiwaPy.sunrisePy import sunrisePy
import time
import numpy as np
from gripper import RobotiqGripper
import matplotlib.pyplot as plt
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
        self.z_min = -0.18      # Z table
        self.z_min_mousse = 12.3

    def getCart(self):
        return self.iiwa.getEEFCartesianPosition()

    def manual_click(self):
        fig = plt.figure()
        connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.imshow(self.camera.color_image)
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

    def from_camera2robot(self, depth_img, u, v, cartpos=None, test_manuel=True):
        '''
        Transform from camera coordinates to robot coordinates
        :param camera: Coordonnées dans le repère caméra (shape : (3, 1) np.array([[x],[y],[z]]))
        :param cartpos: Coordonnées cartésiennes du robot
        :return:
        '''
        if test_manuel:
            u, v = self.manual_click()
        camera = self.camera.transform_3D(u, v, self.camera.depth_image)
        camera = camera.reshape(3, 1)
        print('camera size : {}'.format(camera.shape), camera, np.transpose(camera).shape)
        if cartpos is None:
            cartpos = rob.getCart()
        Base_main = self.camera2main(np.transpose(camera), self.Rmain_camera, self.Tmain_camera)
        R, T = self.get_rotation_translation(cartpos)
        Base_robot = self.fromAtoB(Base_main, np.transpose(R), -T, changeT=False)
        Base_robot = Base_robot.reshape(3)
        print(Base_robot)
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
            speed = 20
            self.iiwa.movePTPLineEEF(goto, speed, orientationVel=0.1)
            OK = input('Continuer ? : ')

    def home(self):
        speed = 20
        home = [595, 15.9, 353, 1.60, 0, 2.77] # Orientation camera vers l'arrière
        goto = [383, -2.13, 329, -1.63, 0, 2.72] # Orientation camera vers l'avant
        self.iiwa.movePTPLineEEF(goto, speed, orientationVel=0.1)

    def grasp(self, pos, ang, speed=40):
        pos[2] = max(pos[2], self.z_min_mousse)
        grasp_above = [pos[0], pos[1], pos[2]+100., ang[0], ang[1], ang[2]]
        self.iiwa.movePTPLineEEF(grasp_above, speed, orientationVel=0.1)
        grasp = [pos[0], pos[1], pos[2], ang[0], ang[1], ang[2]]
        self.iiwa.movePTPLineEEF(grasp, speed, orientationVel=0.1)
        self.grip.closeGripper()
        print(self.grip.isObjectDetected())
        self.iiwa.movePTPLineEEF(grasp_above, speed, orientationVel=0.1)
        self.iiwa.movePTPLineEEF(grasp, speed, orientationVel=0.1)
        self.grip.openGripper()

def onclick(event):
    global x_click, y_click
    x_click, y_click = int(event.xdata), int(event.ydata)
    print(x_click, y_click)

if __name__=="__main__":
    try:
        rob = Robot()
        print(rob.getCart())
        # cart = rob.from_camera2robot(rob.camera.depth_image, 241, 290)
        rob.moveto([[450, -2.13, 329]])
        OK = input('Move à la position cartésienne : {}  Est-ce OK ? (oui : 1)'.format(cart))
        cart = []
        # rob.moveto(cart)
        ang = [-1.5, 0., np.pi]
        rob.grasp(cart, ang)
        print('Position Cartésienne atteinte', rob.getCart())
        rob.camera.stop_pipe()
        rob.iiwa.close()
    except Exception as e:
        print(e)
        rob.camera.stop_pipe()
        rob.iiwa.close()
