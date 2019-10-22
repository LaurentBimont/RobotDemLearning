# Import des classes utiles
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import time as time
import sys, os
import traceback
from os import listdir
from os.path import isfile, join

# Import des différentes classes
import divers as div
from trainer import Trainer
from robot import Robot
from fingertracking import FingerTracker
from camera import RealCamera

# Import de l'argparse
import argparse

def get_pred(trainer, num=None, camera=None, depth=None):
    if (depth is None) and (camera is not None):
        depth_image, _ = camera.get_frame()
        depth = np.copy(depth_image)
    elif depth is not None:
        depth_image = np.copy(depth)
    else:
        print('Erreur dans le get_pred, les arguments donnés ne permettent pas de faire une prédiction\n ')
        print('depth : {}, camera : {}'.format(type(depth), type(camera)))
        return
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
    out_viz = out[:, :, 1]*2 - 1
    viz = False
    x_pred, y_pred, angle_pred, e_pred, pca_zoom = div.postprocess_pred(out)
    x_pred += 6
    y_pred += 6
    if viz:
        rect = div.draw_rectangle(e_pred, angle_pred, x_pred, y_pred, 20)
        optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)
        plt.imshow(out)
        plt.show()
        plt.subplot(1, 2, 1)
        plt.imshow(out)
        plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=4, color='blue')
        plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=4, color='blue')
        plt.scatter(x_pred, y_pred, color='blue')
        # plt.subplot(1, 2, 2)
        # cop_copy_depth = copy_depth[0, :, :, :]
        # cop_copy_depth[cop_copy_depth==0.] = np.max(cop_copy_depth)
        # plt.imshow(cop_copy_depth)
        # plt.scatter(int(x_pred), int(y_pred))
        plt.subplot(1, 2, 2)
        plt.imshow(out)
        plt.show()

    depth_mask = np.copy(depth)
    depth_mask[depth_mask>0.45] = 0
    depth_mask[depth_mask!=0] = 1

    out[:, :, 1] = (out[:, :, 1] - 0.5) * 2
    desired_output = out[:, :, 1]*(out[:, :, 0] != 0).astype(np.int)
    rect = div.draw_rectangle(e_pred, angle_pred, x_pred, y_pred, 20)
    optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)

    plt.imshow(desired_output, vmin=-1, vmax=1, cmap='RdYlGn')
    plt.colorbar(label='Value of Output')

    plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='blue')
    plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='blue')
    plt.scatter(x_pred, y_pred, color='blue')

    if num is not None:
        plt.savefig('figvideo/output_ClefAmpoule{}.png'.format(num), dpi=800)
        plt.show()

    return x_pred, y_pred, angle_pred, e_pred, depth, out

def demo(nb_demo, camera, FT):
    for i in range(1, nb_demo):
        xp_param = []
        x, y = 0, 0
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
        print(label_plt.shape)
        print('Voici ce qui sera enregistré')
        plt.subplot(1, 2, 1)
        plt.imshow(label_plt[:, :, 0])
        plt.show()
        np.save('./Experiences/Demonstration/depth_label/depth_parameters_demo{}.npy'.format(i),
                (depth_image, label_plt))
    return

def learning(demo_depth_label, trainer):
    trainer.main_online(demo_depth_label, nb_batch=1600)
    return trainer

def grasping(nb_trial, trainer, camera, iiwa, camera_param):
    x_pred, y_pred, angle_pred, e_pred, depth, out = get_pred(trainer, camera=camera, num=nb_trial)
    print('Parametre du rectangle : ecartement {}, angle {}, x: {}, y: {}, longueur pince {}'.format(e_pred,
                                                                                                        angle_pred,
                                                                                                        x_pred,
                                                                                                        y_pred,
                                                                                                        20))
    depth = np.concatenate((np.zeros((depth.shape[0], 160)), depth), axis=1)
    target_pos = iiwa.from_camera2robot(depth, int(x_pred+160), int(y_pred), camera_param=camera_param)
    print('Deplacement du robot à : {} avec pour angle {}'.format(target_pos, angle_pred))
    grasp_success = iiwa.grasp(target_pos, angle_pred)
    print('Le grasp a été réussi : ', grasp_success)
    return x_pred, y_pred

def load():
    demo_depth_label = []
    demo_depth_label_file = [join('Experiences/Demonstration/depth_label/', f) for f in
                             listdir('Experiences/Demonstration/depth_label/') if
                             isfile(join('Experiences/Demonstration/depth_label/', f))]
    for f in demo_depth_label_file:
        print('Loading : ', f)
        demo_depth_label.append(np.load(f))
    return demo_depth_label

def action(trainer, Robot, camera, iiwa, camera_param, viz=False):
    '''
    Perform an action decided by a neural network
    :param trainer: Model
    :param Robot: True will perform grasp in real life, False will perform test on automated_test file
    '''
    if Robot:
        cont = '1'
        nb_attempt = 0
        while cont != '0':
            grasping(nb_attempt, trainer, camera, iiwa, camera_param)
            nb_attempt += 1
            cont = input('Voulez vous continuer ? NON:0')
    else:
        for i in range(36):
            depth = np.load('Automated_test/Test_{}.npy'.format(i))
            good_pos = np.load('Automated_test/Good_Point{}.npy'.format(i))
            x_pred, y_pred, _, _, depth, out = get_pred(trainer, depth=depth)
            distance = np.sqrt((good_pos[0]-x_pred)**2 + (good_pos[1]-y_pred)**2)
            print('La distance est de : ', distance)
            if viz:
                plt.subplot(1, 2, 1)
                plt.imshow(depth)
                plt.scatter(good_pos[0], good_pos[1], color='red')
                plt.scatter(x_pred, y_pred, color='black')
                plt.subplot(1, 2, 2)
                viz_out = out[:, :, 1]
                viz_out[viz_out==0] = 0.5
                plt.imshow(viz_out, cmap='RdYlGn', vmin=0, vmax=1)
                plt.show()
    return

def main(args):
    onRobot = args.onRobot
    Training = args.Training
    snapshot_file = args.snapshot_file
    print(snapshot_file)
    Demo = args.Demo
    nb_demo = args.nb_demo
    # nb_demo = False, True, 'Vis_1_Demo_', False, 1
    try:
        if Demo or onRobot:
            camera = RealCamera()
            camera.start_pipe()
            time.sleep(1)
            camera_param = [camera.intr.fx, camera.intr.fy, camera.intr.ppx, camera.intr.ppy, camera.depth_scale]
            FT = FingerTracker()
            time.sleep(1)
            iiwa = Robot()
            iiwa.home()
            # Launch demonstration mode and save in Experiences/depth_label
        if Demo:
            demo(nb_demo, camera, FT)

        # Load previous or present demonstration files
        demo_depth_label = load()

        ### Create a network and train it with previous loaded data
        # If training is set to false, a last weights are loaded into the network
        if Training:
            trainer = Trainer(savetosnapshot=True, load=False, snapshot_file=snapshot_file)
            trainer = learning(demo_depth_label, trainer)
        else:
            trainer = Trainer(savetosnapshot=False, load=True, snapshot_file=snapshot_file)

        ### If Robot is set to True, Grasping will be executed in real life.
        # Else, it will use Automated_test files to make a prediction.
        action(trainer, onRobot, camera, iiwa, camera_param)
        if Demo or onRobot:
            iiwa.iiwa.close()

    except Exception as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        if Demo or onRobot:
            iiwa.iiwa.close()
            camera.stop_pipe()
        del exc_info
        pass

    except RuntimeError as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        if Demo or onRobot:
            iiwa.iiwa.close()
            camera.stop_pipe()

if __name__=="__main__":
    onRobot, Training, snapshot_file, Demo, nb_demo = False, True, 'Vis_1_Demo_', False, 1
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Teach a robot to grasp an object at a precise location from one demonstration')

    parser.add_argument('--onRobot', dest='onRobot', action='store_true', default=True, help='run in simulation?')
    parser.add_argument('--Training', dest='Training', action='store_true', default=False, help='Train the model again ? is set to False, the last model will be loaded')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store', default='default_snapshot_name' ,help='Name of the snapshot file where model will be save')
    parser.add_argument('--Demo', dest='Demo', action='store_true', default=False, help='Perform a demonstration')
    parser.add_argument('--nb_demo', dest='nb_demo', type=int, action='store', default=1, help='number of demonstrations to perform')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
