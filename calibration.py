from camera import RealCamera
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import optimize
from iiwaPy.sunrisePy import sunrisePy

from mpl_toolkits.mplot3d import Axes3D
import time

take_picture = False
Cam = RealCamera()
Cam.start_pipe(usb3=False)
# Parameters of the camera
depth_scale = Cam.depth_scale
fx, fy, Cx, Cy = Cam.intr.fx, Cam.intr.fy, Cam.intr.ppx, Cam.intr.ppy
params = [fx, fy, Cx, Cy, depth_scale]
np.save('intrinsic_parameters.npy', params)

Cam.get_frame()
Cam.stop_pipe()

## 10 : Camera droite par rapport à la table angle Z A
## 11 : Camera droite par rapport à la table angle Z A translation Y et Z
## 12 : Camera droite par rapport à la table angle Z B
## 13 : Camera tordue par rapport à la table

if take_picture:
    ip = '172.31.1.148'
    iiwa = sunrisePy(ip)
    iiwa.attachToolToFlange([-1.5, 1.54, 252.8, 0, 0, 0])
    cart_pos = iiwa.getEEFCartesianPosition()
    iiwa.close()
    Cam = RealCamera()
    Cam.start_pipe(usb3=False)

    # Parameters of the camera
    depth_scale = Cam.depth_scale
    fx, fy, Cx, Cy = Cam.intr.fx, Cam.intr.fy, Cam.intr.ppx, Cam.intr.ppy

    params = [fx, fy, Cx, Cy, depth_scale]
    np.save('intrinsic_parameters.npy', params)

    Cam.get_frame()
    Cam.stop_pipe()
    depth_map = Cam.depth_image
    depth_map[depth_map>1] = 0
    plt.subplot(1, 2, 1)
    plt.imshow(depth_map)
    plt.subplot(1, 2, 2)
    plt.imshow(Cam.color_image)

    plt.show()
    np.save('mycalibcolor13.npy', Cam.color_image)
    np.save('mycalibdepth13.npy', depth_map)
    np.save('robot_cart_pos13.npy', cart_pos)
    print(cart_pos)
    camera_color_img1 = Cam.color_image
    camera_depth_img1 = Cam.depth_image

    print('Fini')

else:
    camera_color_img1 = np.load('calib/mycalibcolor10.npy')
    camera_depth_img1 = np.load('calib/mycalibdepth10.npy')
    cart_pos1 = np.load('calib/robot_cart_pos10.npy')
    camera_color_img2 = np.load('calib/mycalibcolor11.npy')
    camera_depth_img2 = np.load('calib/mycalibdepth11.npy')
    cart_pos2 = np.load('calib/robot_cart_pos11.npy')
    camera_color_img3 = np.load('calib/mycalibcolor12.npy')
    camera_depth_img3 = np.load('calib/mycalibdepth12.npy')
    cart_pos3 = np.load('calib/robot_cart_pos12.npy')
    params = np.load('calib/intrinsic_parameters.npy')

def get_point_from_image(camera_color_img, camera_depth_img, pix1=40, pix2=0, pix3=47):
    checkerboard_size = (8, 6)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)

    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

    checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                              cv2.CALIB_CB_ADAPTIVE_THRESH)

    if checkerboard_found:
        corners = cv2.cornerSubPix(gray_data, corners, (11, 11), (-1, -1), refine_criteria)

        cv2.circle(gray_data, (corners[pix1][0][0], corners[pix1][0][1]), 5, -1)
        cv2.circle(gray_data, (corners[pix2][0][0], corners[pix2][0][1]), 5, -1)
        cv2.circle(gray_data, (corners[pix3][0][0], corners[pix3][0][1]), 5, -1)

        cv2.circle(camera_depth_img, (corners[pix1][0][0], corners[pix1][0][1]), 5, -1)
        cv2.circle(camera_depth_img, (corners[pix2][0][0], corners[pix2][0][1]), 5, -1)
        cv2.circle(camera_depth_img, (corners[pix3][0][0], corners[pix3][0][1]), 5, -1)

        # plt.subplot(1, 2, 1)
        # plt.imshow(gray_data)
        # plt.subplot(1, 2, 2)
        # plt.imshow(camera_depth_img)
        # plt.show()
        P1 = corners[pix1]
        P1_camera = Cam.transform_3D(int(P1[0][0]), int(P1[0][1]), image=camera_depth_img)
        P3 = corners[pix3]
        P3_camera = Cam.transform_3D(int(P3[0][0]), int(P3[0][1]), image=camera_depth_img)
        P2 = corners[pix2]
        P2_camera = Cam.transform_3D(int(P2[0][0]), int(P2[0][1]), image=camera_depth_img)
        return P1_camera, P2_camera, P3_camera

P1_camera_1, P2_camera_1, P3_camera_1 = get_point_from_image(camera_color_img1, camera_depth_img1)
P1_camera_2, P2_camera_2, P3_camera_2 = get_point_from_image(camera_color_img2, camera_depth_img2)
P1_camera_3, P2_camera_3, P3_camera_3 = get_point_from_image(camera_color_img3, camera_depth_img3, pix1=7, pix2=0, pix3=47)

P1_robot, P2_robot, P3_robot = np.array([400.67, 149.05, -11.77]), \
                               np.array([405.11, 260.18, -11.56]),\
                               np.array([556.35, 143.16, -10.94])       # Position des points dans le repère robot

P_robot = np.transpose(np.array([P1_robot, P2_robot, P3_robot]))

### Recherche de la matrice de rotation/Translation  repère Robot au repère Main
def get_rotation(cart_pos):
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

## Position des 3 points dans les deux orientations caméra

camera1 = np.transpose(np.array([P1_camera_1, P2_camera_1, P3_camera_1]))
camera2 = np.transpose(np.array([P1_camera_2, P2_camera_2, P3_camera_2]))
camera3 = np.transpose(np.array([P1_camera_3, P2_camera_3, P3_camera_3]))

##### Conversion pour passer de la base robot à la base poignet
# Calcul des matrices de rotation et translation
R1, T1 = get_rotation(cart_pos1)
R2, T2 = get_rotation(cart_pos2)
R3, T3 = get_rotation(cart_pos3)


def fromAtoB(PA, Ra2b, Ta2b_basea, changeT=True):
    '''
    Fait les calculs suivants : Pb = Ra2b.T x Pa - Ra2b.T x Ta2b_basea
    :param PA: Coordonné d'un point dans le repère A
    :param Ra2b: Rotation de A vers B
    :param Ta2b_basea: Vecteur de translation OaOb exprimé dans la base A
    :return: PB : Coordonnées du point P dans le repère B
    '''
    if changeT:
        T = np.transpose(Ra2b).dot(np.transpose(Ta2b_basea))
    else:       # Cas ou l'utilisateur rentre directement la bonne translation
        T = Ta2b_basea

    print('This is T', T)
    Ta2b_baseb = np.transpose(np.tile(T, (PA.shape[1], 1)))
    print(PA.shape, Ra2b.shape, Ta2b_baseb.shape)
    PB = np.transpose(Ra2b).dot(PA) - Ta2b_baseb
    return PB

# #Préparation de T sous forme de matrice
# T1 = np.transpose(np.tile(T1, (3, 1)))
# T2 = np.transpose(np.tile(T2, (3, 1)))
# T3 = np.transpose(np.tile(T3, (3, 1)))

## Transformation du repère robot au repère poignet
# P_main_1 = np.transpose(R1).dot(P_robot) - T1
# P_main_2 = np.transpose(R2).dot(P_robot) - T2
# P_main_3 = np.transpose(R3).dot(P_robot) - T3
P_main_1 = fromAtoB(P_robot, R1, T1)
P_main_2 = fromAtoB(P_robot, R2, T2)
P_main_3 = fromAtoB(P_robot, R3, T3)

##### Rescaling des nuages de points pour qu'ils aient les mêmes dimensions CADUC #####
camera = np.concatenate((camera1, camera2), axis=1)
P_main = np.concatenate((P_main_1, P_main_2), axis=1)

##### Méthode Mathématiques pour trouver la rotation / translation
# Calcul de la matrice N
def get_rotation_SVD(PCA, PCB):
    # On ramène le nuage de point camera à son centroid
    PCA_centroid = np.transpose(np.tile((PCA.mean(axis=1)), (3, 1)))
    PCA_centre = PCA - PCA_centroid

    # On ramène le nuage de point du end effector à son centroid
    PCB_centroid = np.transpose(np.tile((PCB.mean(axis=1)), (3, 1)))
    PCB_centre = PCB - PCB_centroid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PCA_centre[0, :], PCA_centre[1, :], PCA_centre[2, :], c='g', marker='^')
    ax.scatter(PCB_centre[0, :], PCB_centre[1, :], PCB_centre[2, :], c='b', marker='^')
    ax.set_title('Nuages centrés')
    plt.show()

    N = PCA_centre.dot(np.transpose(PCB_centre))
    U, S, Vt = np.linalg.svd(N)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1

    PCA_centre_rotation = R.dot(PCA_centre)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PCA_centre_rotation[0, :], PCA_centre_rotation[1, :], PCA_centre_rotation[2, :], c='g', marker='^')
    ax.scatter(PCB_centre[0, :], PCB_centre[1, :], PCB_centre[2, :], c='b', marker='^')
    ax.set_title('Rotation censée marcher')
    plt.show()

    print(PCA.mean(axis=1), PCA.mean(axis=1).shape)
    T = -R.dot(PCA.mean(axis=1)) + PCB.mean(axis=1)
    return R, T

def applyrotation(PCA, PCB, R, T):
    # On ramène le nuage de point camera à son centroid
    PCA_centroid = np.transpose(np.tile((PCA.mean(axis=1)), (3, 1)))
    PCA_centre = PCA - PCA_centroid
    PCA_centre_rotation = R.dot(PCA_centre)

    PCB_centroid = np.transpose(np.tile((PCB.mean(axis=1)), (3, 1)))
    PCB_centre = PCB - PCB_centroid

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PCA_centre_rotation[0, :], PCA_centre_rotation[1, :], PCA_centre_rotation[2, :], c='g', marker='^')
    ax.scatter(PCB_centre[0, :], PCB_centre[1, :], PCB_centre[2, :], c='b', marker='^')
    ax.set_title('Rotation censée marcher')
    plt.show()

R, T = get_rotation_SVD(P_main_2, camera2)
np.save('CalibrationRotation.npy', R)
np.save('CalibrationTranslation.npy', T)

# Test sur le nuage 1
applyrotation(P_main_1, camera1, R, T)
applyrotation(P_main_2, camera2, R, T)
applyrotation(P_main_3, camera3, R, T)

P1_cam = fromAtoB(P_main_1, R, T)
print('Ultime test', P1_cam, camera1)
P2_cam = fromAtoB(P_main_2, R, T)
print('Ultime test', P2_cam, camera2)
P3_cam = fromAtoB(P_main_3, R, T)
print('Ultime test', P3_cam, camera3)

P2_cam = R.dot(P_main_2) + np.transpose(np.tile(T, (3, 1)))

print(P2_cam, np.transpose(np.tile(T, (3, 1))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P2_cam[0, :], P2_cam[1, :], P2_cam[2, :], c='g', marker='^')
ax.scatter(camera2[0, :], camera2[1, :], camera2[2, :], c='b', marker='^')
ax.set_title('Test ultime')
plt.show()


#### Camera to Robot
# Camera to main

def camera2main(camera, R, T):
    print(camera.shape)
    return np.transpose(R).dot(np.transpose(camera) - np.transpose(np.tile(T, (camera.shape[0], 1))))

# camerainmain = camera2main(camera2, R, T)
# camerainrobot = fromAtoB(camerainmain, np.transpose(R2), -T2, changeT=False)

def fromcamera2robot(camera, Rmain_camera, Tmain_camera, cartpos):
    Base_main = camera2main(camera, Rmain_camera, Tmain_camera)
    R, T = get_rotation(cartpos)
    Base_robot = fromAtoB(Base_main, np.transpose(R), -T, changeT=False)
    return Base_robot

print(camera1[:,:1].shape, camera1[:,:1])

camerainrobot = fromcamera2robot(np.transpose(camera2[:, :1]), R, T, cart_pos2)
print(P_robot, '\n\n', camerainrobot)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P_robot[0, 0], P_robot[1, 0], P_robot[2, 0], c='g', marker='^')
ax.scatter(camerainrobot[0, :], camerainrobot[1, :], camerainrobot[2, :], c='b', marker='^')
ax.set_title('Vraiment l\'ultime test')
plt.show()
