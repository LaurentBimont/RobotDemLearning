import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from skimage.transform import resize
import time as time
import sys, os
import traceback

## Import des différentes classes
import divers as div
from trainer import Trainer
from robot import Robot
from fingertracking import FingerTracker
from camera import RealCamera

#########Fonction pour la lisbilité #########
def get_pred(camera, trainer):
    depth_image, _ = camera.get_frame()
    depth = np.copy(depth_image)

    init_shape = depth_image.shape
    depth_image = div.preprocess_depth_img(depth_image)
    depth_image = resize(depth_image, (224, 224, 3), anti_aliasing=True)
    depth_image = depth_image.reshape((1, 224, 224, 3))
    output_prob = trainer.forward(depth_image)
    out = trainer.prediction_viz(output_prob, depth_image)
    print('Taille de sortie : ', out.shape)
    out = out.reshape((224, 224, 3))
    out = resize(out, init_shape)
    plt.imshow(out)
    plt.show()
    np.save('outfrofrancois.npy', out)
    x_pred, y_pred, angle_pred, e_pred = div.postprocess_pred(out)
    viz = True
    if viz:
        rect = div.draw_rectangle(e_pred, angle_pred, x_pred, y_pred, 20)
        optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)
        plt.subplot(1, 3, 1)
        plt.imshow(out)
        plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='yellow')
        plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='yellow')
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(_, cv2.COLOR_BGR2RGB))
        # plt.savefig('pred{}.png'.format(i), dpi=600)
        # if execute=='1':
        plt.subplot(1, 3, 3)
        plt.imshow(depth)
        plt.scatter(int(x_pred), int(y_pred))
        plt.show()
    return x_pred, y_pred, angle_pred, e_pred, depth

######### Initialisation des différents outils #########
# iiwa = Robot()
camera = RealCamera()  # Vraiment utile ?
# FT = FingerTracker()
time.sleep(1)
######### Mise en Position #########
# iiwa.home()
######### Lancement de FingerTracker #########


######### Lancement de Trainer #########
# retrain = False
# if retrain:
#     FingerTrack = input('Voulez-vous apprendre au robot ? oui=1')
#     if FingerTrack:
#         x, y, angle, e, lp, depth_image, color_image = FT.main()
#         trainer = Trainer()
#         trainer.main([x, y, angle, e, lp], depth_image)
# else:
#     trainer = Trainer(load=True, snapshot_file='reference_good')

trainer = Trainer(load=True, snapshot_file='reference_good')

######### Prédiction #########
# camera.start_pipe()
# time.sleep(1)
# camera_param = [camera.intr.fx, camera.intr.fy, camera.intr.ppx, camera.intr.ppy, camera.depth_scale]
# Continue = '1'
# try:
#     while Continue == '1':
#         x_pred, y_pred, angle_pred, e_pred, depth = get_pred(camera, trainer)
#         print('Parametre du rectangle : ecartement {}, angle {}, x: {}, y: {}, longueur pince {}'.format(e_pred, angle_pred, x_pred, y_pred, 20))
#         target_pos = iiwa.from_camera2robot(depth, int(x_pred), int(y_pred), camera_param=camera_param)
#         print('Deplacement du robot à : {} avec pour angle {}'.format(target_pos, angle_pred))
#         grasp_success = iiwa.grasp(target_pos, angle_pred)
#
#         Continue = input('Encore ? oui=1')
#
# except Exception as e:
#     exc_info = sys.exc_info()
#     traceback.print_exception(*exc_info)
#     del exc_info
#     pass
####### Clean Zone #######

camera.start_pipe()
time.sleep(1)
camera_param = [camera.intr.fx, camera.intr.fy, camera.intr.ppx, camera.intr.ppy, camera.depth_scale]

demo, trial = 0, 0
# try:
while True:
    DO = input('What do you want to do ? (Demo : 1), (Retrained : 2), (grasp : 3), (stop : 4)')
    if DO == '1':
        redo = 0
        while redo == 0:
            x, y, angle, e, lp, depth_image, _ = FT.main(camera)
            redo = input('Keep that demo ? (Non : 0), (Oui : 1)')
        depth_image = div.preprocess_depth_img(depth_image)
        np.save('Experiences/depth_demo{}.npy'.format(demo), depth_image)
        np.save('Experiences/parameters_demo{}.npy'.format(demo), np.array([x, y, angle, e, lp]))
        demo += 1

    if DO == '2':
        trainer.exp_rpl.clean()
        print('Experience Replay reset is finished')
        depth_demo, param_demo = [], []

        for i in range(demo+1):
            depth_demo.append(np.load('Experiences/depth_demo{}.npy'.format(i)))
            param_demo.append(np.load('Experiences/parameters_demo{}.npy'.format(i)))

            trainer.main_without_backprop(np.load('Experiences/depth_demo{}.npy'.format(i)),
                                          np.load('Experiences/parameters_demo{}.npy'.format(i)),
                                          augmentation_factor=3)
        depth_trial, param_trial = [], []

        if trial != 0:
            for i in range(trial+1):
                depth_trial.append(np.load('Experiences/depth_trial{}.npy'.format(i)))
                param_trial.append(np.load('Experiences/parameters_trial{}.npy'.format(i)))
                trainer.main_without_backprop(np.load('Experiences/depth_trial{}.npy'.format(i)),
                                              np.load('Experiences/parameters_trial{}.npy'.format(i)),
                                              augmentation_factor=3)
        print('starting main training')
        ### Create experience replay ranking
        # Train with experience replay
        trainer.main_xpreplay(batch_size=1)

    if DO == '3':
        x_pred, y_pred, angle_pred, e_pred, depth = get_pred(camera, trainer)
        print('Parametre du rectangle : ecartement {}, angle {}, x: {}, y: {}, longueur pince {}'.format(e_pred,
                                                                                                         angle_pred,
                                                                                                         x_pred,
                                                                                                         y_pred,
                                                                                                         20))
        target_pos = iiwa.from_camera2robot(depth, int(x_pred), int(y_pred), camera_param=camera_param)
        print('Deplacement du robot à : {} avec pour angle {}'.format(target_pos, angle_pred))
        grasp_success = iiwa.grasp(target_pos, angle_pred)
        print(grasp_success)
        ####### ADD SAVING ######
    if DO == '4':
        break
#
# except Exception as e:
#     exc_info = sys.exc_info()
#     traceback.print_exception(*exc_info)
#     del exc_info
#     pass

# iiwa.iiwa.close()
camera.stop_pipe()

########## Execution ##########

# plt.subplot(2, 2, 1)
# plt.imshow(optim_rectangle)
# plt.subplot(2, 2, 2)
# plt.imshow(out[:, :, 1] * optim_rectangle)
# plt.subplot(2, 2, 3)
# plt.imshow(out)
# plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='yellow')
# plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='yellow')
# plt.subplot(2, 2, 4)
# out[:, :, 0] = out[:, :, 0] * optim_rectangle
# out[:, :, 1] = out[:, :, 1] * optim_rectangle
# plt.imshow(out)
# plt.show()
# Pour les angles : voir dans divers.py

# model = Trainer(load=True, snapshot_file='reference_good')
# im = np.zeros((1, 224, 224, 3), np.float32)
# im[:, 70:190, 100:105, :] = 1
# im[:, 70:80, 80:125, :] = 1
# im[:, 150:160, 150:160, :] = 1
# im = tf.contrib.image.rotate(im, angles=45)
# model.forward(im)
#
# out[out < 0] = 0
# zoom_pixel = 30
#
# (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
# test_pca = out[y_max-zoom_pixel:y_max+zoom_pixel, x_max-zoom_pixel:x_max+zoom_pixel, 1]
# PointCloud = div.heatmap2pointcloud(test_pca)
# pca = PCA()
# pca.fit(PointCloud)
# vectors = pca.components_
# sing_val = pca.singular_values_/np.linalg.norm(pca.singular_values_)
# vectors[0] *= sing_val[0]
# vectors[1] *= sing_val[1]
# np.linalg.norm(pca.singular_values_)
# origin = [zoom_pixel], [zoom_pixel]
# e = 30
# theta = div.py_ang([1, 0], vectors[0])*180/np.pi


# rect = div.draw_rectangle(e, theta, x_max, y_max, 20)
# optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)


# plt.subplot(2, 2, 1)
# plt.imshow(optim_rectangle)
# plt.subplot(2, 2, 2)
# plt.imshow(out[:, :, 1] * optim_rectangle)
# plt.subplot(2, 2, 3)
# plt.imshow(out)
# plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='yellow')
# plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='yellow')
# plt.subplot(2, 2, 4)
# out[:, :, 0] = out[:, :, 0] * optim_rectangle
# out[:, :, 1] = out[:, :, 1] * optim_rectangle
# plt.imshow(out)
# plt.show()

# if __name__=="__main__":

    # test_rectangle = div.draw_rectangle(20, 20, 50, 150, 60)
    # print(test_rectangle)
    # autre_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), test_rectangle, color=1)

