import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from skimage.transform import resize
import time as time
import sys, os
import traceback
from os import listdir
from os.path import isfile, join


## Import des différentes classes
import divers as div
from trainer import Trainer
from robot import Robot
from fingertracking import FingerTracker
from camera import RealCamera
import experiment
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
    out = out.reshape((224, 224, 3))
    out = resize(out, init_shape)
    print(out)
    print(np.max(out))
    viz = True
    x_pred, y_pred, angle_pred, e_pred = div.postprocess_pred(out,  camera)

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
iiwa = Robot()
camera = RealCamera()  # Vraiment utile ?
FT = FingerTracker()
time.sleep(1)
######### Mise en Position #########
iiwa.home()

trainer = Trainer(load=False, snapshot_file='28juin')

####### Clean Zone #######
camera.start_pipe()
time.sleep(1)
camera_param = [camera.intr.fx, camera.intr.fy, camera.intr.ppx, camera.intr.ppy, camera.depth_scale]

### Paramètre à toucher ###
demo, trial = 0, 0
load = True

###########################
demo_depth, demo_label = [], []
explo_depth, explo_label = [], []
try:
    if load:
        demoFileDepth = [join('Experiences/Demonstration/depth/', f) for f in listdir('Experiences/Demonstration/depth/') if
                         isfile(join('Experiences/Demonstration/depth/', f))]
        demoFileLabel = [join('Experiences/Demonstration/label/', f) for f in listdir('Experiences/Demonstration/label/') if
                         isfile(join('Experiences/Demonstration/label/', f))]
        for f_depth, f_label in zip(demoFileDepth, demoFileLabel):
            demo_depth.append(np.load(f_depth))
            demo_label.append(np.load(f_label))

        exploFileDepth = [join('Experiences/Exploration/depth/', f) for f in listdir('Experiences/Exploration/depth/') if
                         isfile(join('Experiences/Exploration/depth/', f))]
        exploFileLabel = [join('Experiences/Exploration/label/', f) for f in listdir('Experiences/Exploration/label/') if
                         isfile(join('Experiences/Exploration/label/', f))]
        for f_depth, f_label in zip(exploFileDepth, exploFileLabel):
            explo_depth.append(np.load(f_depth))
            explo_label.append(np.load(f_label))

    eef_point = [] 
    demonstration_point = []
    ref_points =[]


    #Find targets on object in the workspace 
    ref_points = FT.detect_blue(camera)
    print(ref_points) 
    depth_image = None

    while True:
        DO = input('What do you want to do ? (Demo : 1), (Retrained : 2), (grasp : 3), (stop : 4)')
        if DO == '1':

            label_plt = [] 

            for point in ref_points:
                continue_demo = '0'
                xp_param = []
                FT.x_ref , FT.y_ref = point 
                while continue_demo == '0':
                    redo = '0'
                    while redo == '0':
                        x, y, angle, e, lp, depth_image, _ = FT.main(camera)
                        redo = input('Keep that demo ? (Non : 0), (Oui : 1)')
                    label_val_ = input('Quelle type de démo ? (bon:1), (mauvais:2)')
                    if label_val_ == '1':
                        label_val = 1
                    else:
                        label_val = -1
                    xp_param.append([x, y, angle, e, lp, label_val])
                    continue_demo = input('Continue demonstrating ? (oui:0) (non:1)')
                
                depth_image = div.preprocess_depth_img(depth_image)
                print(xp_param)
                label_plt.append(div.compute_labels(xp_param, shape=depth_image.shape))
                plt.subplot(1, 2, 1)
                plt.imshow(label_plt[-1])
                plt.subplot(1, 2, 2)
                plt.imshow(depth_image)
                plt.show()
            np.save('Experiences/Demonstration/depth/depth_demo{}.npy'.format(demo), depth_image)
            np.save('Experiences/Demonstration/label/parameters_demo{}.npy'.format(demo), label_plt)
            demo += 1
            demo_depth.append(depth_image)
            demo_label.append(label_plt)

        elif DO == '2':
            quefaire = input('Recalculer la DataFrame ? (oui:1), (non : 2)')
            if quefaire == '1':
                trainer.exp_rpl.clean()
                print('Experience Replay reset is finished')

                ### Create experience replay ranking
                for depth, list_label in zip(demo_depth, demo_label):
                    for label in list_label: 
                        trainer.main_without_backprop(depth,
                                                  label,
                                                  augmentation_factor=3,
                                                  demo=True)
                for depth, label in zip(explo_depth, explo_label):
                    trainer.main_without_backprop(depth,
                                                  label,
                                                  augmentation_factor=3,
                                                  demo=False)
                print('starting main training')
            ### Train with experience replay
            trainer.main_xpreplay(nb_epoch=2, batch_size=1)

        elif DO == '3':
            x_pred, y_pred, angle_pred, e_pred, depth = get_pred(camera, trainer)
            print('Parametre du rectangle : ecartement {}, angle {}, x: {}, y: {}, longueur pince {}'.format(e_pred,
                                                                                                             angle_pred,
                                                                                                             x_pred,
                                                                                                             y_pred,
                                                                                                             20))
            target_pos = iiwa.from_camera2robot(depth, int(x_pred), int(y_pred), camera_param=camera_param)
            print('Deplacement du robot à : {} avec pour angle {}'.format(target_pos, angle_pred))
            grasp_success = iiwa.grasp(target_pos, angle_pred)
            print('Le grasp a été réussi : ', grasp_success)
            if grasp_success:
                label_value = 1
            else:
                label_value = -1
            depth_image = div.preprocess_depth_img(depth)
            depth_image = resize(depth_image, (224, 224, 3), anti_aliasing=True)
            label_plt = div.compute_labels([[x_pred, y_pred, angle_pred, 0.5*e_pred, 0.5*1.2*e_pred, label_value]])
            np.save('Experiences/Exploration/depth/depth_exploration{}.npy'.format(trial), depth_image)
            np.save('Experiences/Exploration/label/parameters_exploration{}.npy'.format(trial), label_plt)
            explo_depth.append(depth_image)
            explo_label.append(label_plt)
            trial += 1
        elif DO == '4':
            break

except Exception as e:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    del exc_info
    pass
finally: 
    iiwa.iiwa.close()
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

