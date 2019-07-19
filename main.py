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

#########Fonction pour la lisbilité #########

def get_pred(trainer):
    depth_image, _ = camera.get_frame()
    depth = np.copy(depth_image)

    init_shape = depth_image.shape
    depth_image = div.preprocess_depth_img(depth_image)
    depth_image = resize(depth_image, (224, 224, 3), anti_aliasing=True)
    depth_image = (depth_image * 255).astype('uint8')

    depth_image = depth_image.reshape((1, 224, 224, 3))

    copy_depth = depth_image.copy()
    output_prob = trainer.forward(depth_image)
    out_numpy = output_prob[0].numpy()
    

    out = trainer.prediction_viz(out_numpy, depth_image)

    out = out.reshape((224, 224, 3))
    out = resize(out, init_shape)

    viz = True
    x_pred, y_pred, angle_pred, e_pred, pca_zoom = div.postprocess_pred(out)

    if viz:
        rect = div.draw_rectangle(e_pred, angle_pred, x_pred, y_pred, 20)
        optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)
        plt.subplot(2, 2, 1)
        plt.imshow(out)
        plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='yellow')
        plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='yellow')
        plt.subplot(2, 2, 2)
        plt.imshow(pca_zoom)
        #plt.imshow(cv2.cvtColor(_, cv2.COLOR_BGR2RGB))
        # plt.savefig('pred{}.png'.format(i), dpi=600)
        # if execute=='1':
        plt.subplot(2, 2, 3)
        plt.imshow(copy_depth[0, :, :, :])
        # plt.scatter(int(x_pred), int(y_pred))
        plt.subplot(2, 2, 3)
        plt.imshow(out[:, :, 1])
        plt.show()
    return x_pred, y_pred, angle_pred, e_pred, depth

def demo(nb_demo):
    xp_param = []
    x, y = 0,0
    keep, other_point = '0', '1'
    while other_point == '1':
        while keep == '0':
            x, y, angle, e, lp, depth_image, _ = FT.main(camera)
            keep = input('Keep that demo ? (Non : 0), (Oui : 1)')
        label_val_ = input('Quelle type de démo ? (bon:1), (mauvais:2)')
        if label_val_ == '1':
            label_val = 1
        else:
            label_val = -1

        xp_param.append([x, y, angle, e, lp, label_val])
        print([x, y, angle, e, lp, label_val])
        other_point = input('Another demonstration on the same object ?(Non : 0), (Oui : 1)')
        keep = '0'

    depth_image = div.preprocess_depth_img(depth_image)
    label_plt = div.compute_labels(xp_param, shape=depth_image.shape)
    print('Voici ce qui sera enregistré')
    plt.subplot(1, 2, 1)
    plt.imshow(label_plt[:, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(depth_image)
    plt.show()
    np.save('./Experiences/Demonstration/depth_label/depth_parameters_demo{}.npy'.format(nb_demo), (depth_image, label_plt))

    demo_depth_label.append((depth_image, label_plt))
    return x, y

def learning(demo_depth_label, explo_depth_label, trainer):
    quefaire = input('Recalculer la DataFrame ? (oui:1), (non : 2)')
    if quefaire == '1':
        trainer.exp_rpl.clean()
        print('Experience Replay reset is finished')

        ### Create experience replay ranking
        for depth, label in demo_depth_label:
            trainer.main_without_backprop(depth,
                                            label,
                                            augmentation_factor=2,
                                            demo=True)
        for depth, label in explo_depth_label:
            trainer.main_without_backprop(depth,
                                            label,
                                            augmentation_factor=2,
                                            demo=False)
            print('starting main training')
        ### Train with experienceReplay replay
    trainer.main_xpreplay(nb_epoch=1 , batch_size=1)
    return trainer

def viz_grap(trainer):

    x_pred, y_pred, angle_pred, e_pred, depth = get_pred(trainer)

def grasping(nb_trial):
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
    depth_image = resize(depth_image, (224, 224, 3), order=2, anti_aliasing=True)
    label_plt = div.compute_labels([[x_pred, y_pred, angle_pred, 0.5*e_pred, 0.5*1.2*e_pred, label_value]])
    np.save('Experiences/Exploration/depth_label/depth_parameters_exploration{}.npy'.format(nb_trial), (depth_image, label_plt))

    explo_depth_label.append((depth_image, label_plt))
    return x_pred, y_pred

def load():
    demo_depth_label_file = [join('Experiences/Demonstration/depth_label/', f) for f in
                             listdir('Experiences/Demonstration/depth_label/') if
                             isfile(join('Experiences/Demonstration/depth_label/', f))]
    for f in demo_depth_label_file:
        demo_depth_label.append(np.load(f))

    explo_depth_label_file = [join('Experiences/Exploration/depth_label/', f) for f in
                              listdir('Experiences/Exploration/depth_label/') if
                              isfile(join('Experiences/Exploration/depth_label/', f))]
    for f in explo_depth_label_file:
        explo_depth_label.append(np.load(f))
    return demo_depth_label, explo_depth_label

def proj_dist(p1, p2):
    d = p1[:2]-p2[:2]
    d = np.sqrt(np.dot(d, d))
    return d

def test(trainer, isdemo, isgrasp, istrain, isreload, ):
    nb_demo, nb_trial = 0, 0
    demo_depth_label = []
    if isreload:
        demo_depth_label, explo_depth_label = load()
    if isdemo:
        _, _ = demo(nb_demo)
        nb_demo += 1
        demo_depth_label, explo_depth_label = load()
    if istrain:
        trainer = learning(demo_depth_label, explo_depth_label, trainer)
    if isgrasp:
        cont = '1'
        nb_attempt = 0
        while cont=='1':
            viz_grap(trainer)
            # grasping(nb_attempt)
            nb_attempt += 1
            cont = input('Voulez vous continuer ? ')
        # grasping(nb_trial)

def validation(camera):
    nb_demo = 0 
    nb_trial = 0 

    while True: 
        try:
            ref_point = FT.detect_blue(camera)[0]
            ref_point3D = camera.transform_3D(*ref_point)

            for i in range(2): 
                demo_point = demo(nb_demo, demo_depth_label)
                demo_point3D = camera.transform_3D(*demo_point)
                nb_demo += 1
            d1 = proj_dist(ref_point3D, demo_point3D)
            print("distance ref demo : {} ".format(d1))

            learning(demo_depth_label, explo_depth_label)

            for i in range(4):
                ref_point = FT.detect_blue(camera)[0]
                ref_point3D = camera.transform_3D(*ref_point)

                decision_point = grasping(nb_trial)
                nb_trial+=1
                decision_point3D = camera.transform_3D(*decision_point)

                d2 = proj_dist(ref_point3D, decision_point3D)
                with open("distance.csv","w+") as f: 
                    print("distance ref demo : {} ".format(d1))
                    print("distance ref decision : {} ".format(d2)) 
                    line = ";".join(map(str, [d1,d2,abs(d1-d2),"\n"])) 
                    f.write(line)
        except KeyboardInterrupt:
            print("fin programme")

if __name__=="__main__":
    ######### Initialisation des différents outils #########
    # iiwa = Robot()
    ######### Mise en Position #########
    # iiwa.home()

    trainer = Trainer(savetosnapshot=True, load=False, snapshot_file='ampouletanh')

    ####### Clean Zone #######

    ### Paramètre à toucher ###
    load_needed = False

    demo_depth_label = []
    explo_depth_label = []
    try:
        if load_needed:
            demo_depth_label_file = [join('Experiences/Demonstration/depth_label/', f) for f in listdir('Experiences/Demonstration/depth_label/') if
                            isfile(join('Experiences/Demonstration/depth_label/', f))]
            for f in demo_depth_label_file:
                demo_depth_label.append(np.load(f))

            explo_depth_label_file = [join('Experiences/Exploration/depth_label/', f) for f in listdir('Experiences/Exploration/depth_label/') if
                                            isfile(join('Experiences/Exploration/depth_label/', f))]
            for f in explo_depth_label_file:
                explo_depth_label.append(np.load(f))

        test(trainer, isdemo=False, isgrasp=True, istrain=True, isreload=True)
        # validation(camera)

    except Exception as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        pass
    except RuntimeError as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)

        # iiwa.iiwa.close()

