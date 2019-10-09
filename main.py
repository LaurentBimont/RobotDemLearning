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

def demo(nb_demo):
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

def grasping(nb_trial):
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

def action(trainer, Robot, viz=False):
    '''
    Perform an action decided by a neural network
    :param trainer: Model
    :param Robot: True will perform grasp in real life, False will perform test on automated_test file
    '''
    if Robot:
        cont = '1'
        nb_attempt = 0
        while cont != '0':
            grasping(nb_attempt)
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
    onRobot, Training, snapshot_file, Demo, nb_demo = False, True, 'Vis_1_Demo_', False, 1
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
            demo(nb_demo)

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
        # Else, it will use Automated_test files to make predictions.
        action(trainer, onRobot)
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

    # --------------- Setup options ---------------
    parser.add_argument('--onRobot', dest='onRobot', action='store_true', default=False, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,
                        help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,
                        help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,
                        help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,
                        help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',
                        help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,
                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',
                        default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,
                        help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,
                        help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,
                        help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,
                        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
