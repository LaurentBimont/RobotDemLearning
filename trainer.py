# -*- coding: utf-8 -*-

import model as mod
import tensorflow as tf
import time
import tqdm
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)

import rewardManager as RM
import functools as func
import numpy as np
import matplotlib.pyplot as plt
import divers as div
import cv2
import scipy as sc
import dataAugmentation as da
import tfmpl                             # Put matplotlib figures in tensorboard
import os
from skimage.transform import resize
from experienceReplay import ExperienceReplay

class Trainer(object):
    def __init__(self, savetosnapshot=True, load=False, snapshot_file='name'):
        super(Trainer, self).__init__()
        self.myModel = mod.Reinforcement()
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
        self.width, self.height = 0, 0
        self.best_idx, self.future_reward = [0, 0], 0
        self.scale_factor = 3
        self.output_prob = 0
        self.loss_value = 0
        self.iteration = 0
        self.num = 0
        self.classifier_boolean = False
        self.savetosnapshot = savetosnapshot
        # self.create_log()

        # Frequency
        self.viz_frequency = 200
        self.saving_frequency = 199
        self.dataAugmentation = da.OnlineAugmentation()
        # Initiate logger
        self.create_log()
        checkpoint_directory = "checkpoint_1"

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.myModel,
                                              optimizer_step=tf.train.get_or_create_global_step())
        if savetosnapshot:
            self.snapshot_file = os.path.join(checkpoint_directory, snapshot_file)
            self.checkpoint.save(self.snapshot_file)
            print('Creating snapshot : {}'.format(self.snapshot_file))
        if load:
            latest_snapshot_file = tf.train.latest_checkpoint(checkpoint_directory)
            status = self.checkpoint.restore(latest_snapshot_file)
            print('Pre-trained model snapshot loaded from: {}'.format(latest_snapshot_file))

        self.loss = tf.losses.huber_loss

        # Initialize the network with a fake shot
        self.forward(np.zeros((1, 224, 224, 3), np.float32))
        self.vars = self.myModel.trainable_variables

    def forward(self, input):
        self.image = input
        # Increase the size of the image to have a relevant output map
        input = div.preprocess_img(input, target_height=self.scale_factor*224, target_width=self.scale_factor*224)
        # Pass input data through model
        self.output_prob = self.myModel(input)
        self.batch, self.width, self.height = self.output_prob.shape[0], self.output_prob.shape[1], self.output_prob.shape[2]
        # Return Q-map
        return self.output_prob

    def compute_loss_dem(self, label, noBackprop=False):
        for l, output in zip(label, self.output_prob):
            l = self.reduced_label(l)[0]

            l_numpy = l.numpy()
            l_numpy_pos = l_numpy[:, :, 0]
            l_numpy_pos[l_numpy_pos > 10] = -1
            l_numpy[:, :, 0] = l_numpy_pos
            l = tf.convert_to_tensor(l_numpy)

            ### Test Sans poids
            weight = np.abs(output.numpy())
            weight[l_numpy > 0] += 20/(np.sqrt(np.sum(l_numpy > 0)+1))       #Initialement 2.
            weight[l_numpy < 0] += 20/(np.sqrt(np.sum(l_numpy < 0)+1))       #Initialement 1.
            weight[l_numpy == 0] += 1/(np.sqrt(np.sum(l_numpy == 0)+1))      #Initialement 0.2

            # print(1/(np.log(np.sum(l_numpy == 0)+2)), 1/(np.log(np.sum(l_numpy > 0)+2)))
            # print('Les poids : ', 1/(np.sum(l_numpy > 0)+1), 1/(np.sum(l_numpy < 0)+1), 1/(np.sum(l_numpy == 0)+1))

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars])

            self.loss_value = self.loss(l, output, weight) + 0.00005 * lossL2
            # print('Contribution pour chaque classe \n 1 {} \n -1 {} \n 0 {}'.format(10/(np.sum(l_numpy > 0)+1), 10/(np.sum(l_numpy < 0)+1), 1/(np.sum(l_numpy == 0)+1)))

            new_lab = l.numpy()
            new_lab = new_lab.reshape((1, *new_lab.shape))

        if (tf.train.get_global_step() is not None) \
                and (tf.train.get_global_step().numpy() % self.viz_frequency == 0) \
                and (not noBackprop):
            print('Printing to Tensorboard')
            print('Contribution des loss : Huber ({}) et L2 reg ({})'.format(
                self.loss(l, output, weight) / self.loss_value,
                0.00005 * lossL2 / self.loss_value))
            img_tensorboard = self.prediction_viz(self.output_prob[0], self.image)
            img_tensorboard_target = self.prediction_viz(new_lab[0, :, :, :], self.image)
            subplot_viz = self.draw_scatter_subplot(img_tensorboard, img_tensorboard_target)
            self.log_fig('subplot_viz', subplot_viz)
            self.log_img('input', self.image)
            self.log_scalar('loss value_dem', self.loss_value)
        # Saving a snapshot file
        if (self.savetosnapshot)\
                and (tf.train.get_global_step() is not None)\
                and (tf.train.get_global_step().numpy()%self.saving_frequency == 0)\
                and (not noBackprop):

            self.save_model()
        return self.loss_value

    def compute_labels(self, label_value, best_pix_ind, shape=(224,224,3), viz=True):
        '''Create the targeted Q-map
        :param label_value: Reward of the action
        :param best_pix_ind: (Rectangle Parameters : x(colonne), y(ligne), angle(en degré), ecartement(en pixel)) Pixel where to perform the action
        :return: label : an 224x224 array where best pix is at future reward value
                 label_weights : a 224x224 where best pix is at one
        '''
        # Compute labels
        x, y, angle, e, lp = best_pix_ind
        rect = div.draw_rectangle(e, angle, x, y, lp)
        label = np.zeros(shape, dtype=np.float32)

        cv2.fillConvexPoly(label, rect, color=1)
        label *= label_value

        if viz:
            plt.subplot(1, 2, 1)
            self.image = np.reshape(self.image, (self.image.shape[1], self.image.shape[2], 3))
            plt.imshow(self.image)
            plt.subplot(1, 2, 2)
            label_viz = np.reshape(label, (224, 224, 3))
            plt.imshow(label)
            plt.show()
        return label

    def reduced_label(self, label):
        '''Reduce label Q-map to the output dimension of the network
        :param label: 224x224 label map
        :return: label and label_weights in output format
        '''

        label = tf.convert_to_tensor(label, np.float32)
        label = tf.image.resize_images(label, (self.width, self.height))
        label = tf.reshape(label[:, :, 0], (self.batch, self.width, self.height, 1))

        if self.classifier_boolean:
            label = label.numpy()
            label[label > 0.] = 1
            label = tf.convert_to_tensor(label, np.float32)
        return label

    def main_batches(self, im, label):
        self.future_reward = 1
        # On change le label pour que chaque pixel sans objet soit considéré comme le background
        if type(im).__module__ != np.__name__:
            im_numpy, label_numpy = im.numpy(), label.numpy()
        else:
            im_numpy, label_numpy = im, label
        mask = (im_numpy>20).astype(np.int32)
        label_numpy = label_numpy * mask
        label = tf.convert_to_tensor(label_numpy, dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.myModel.trainable_variables)
            self.forward(im)
            self.compute_loss_dem(label)
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())
            self.iteration = tf.train.get_global_step()

    def main_online(self, demo_depth_label, nb_batch=1600):
        self.dataAugmentation.create_database(demo_depth_label)
        for batch in tqdm.tqdm(range(nb_batch)):
            t0 = time.time()
            batch_im, batch_lab = self.dataAugmentation.get_pair()
            self.main_batches(tf.stack([batch_im]), tf.stack([batch_lab]))

#### Accessoires #######
    def vizualisation(self, img, idx):
        prediction = cv2.circle(img[0], (int(idx[1]), int(idx[0])), 7, (255, 255, 255), 2)
        plt.imshow(prediction)
        plt.show()

    def save_model(self):
        print("Saving model to {}".format(self.snapshot_file))
        self.checkpoint.save(self.snapshot_file)

    def prediction_viz(self, qmap, im):
        qmap = (qmap + 1) / 2
        qmap = tf.image.resize_images(qmap, (self.width, self.height))
        qmap = tf.image.resize_images(qmap, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        qmap = tf.reshape(qmap[:, :, 0], (224, 224))

        x_map, y_map = np.argmax(np.max(qmap, axis=1)), np.argmax(np.max(qmap, axis=0))
        rescale_qmap = qmap
        img = np.zeros((224, 224, 3))
        img[:, :, 0] = im[0, :, :, 0] / np.max(im[0, :, :, 0])
        img[:, :, 1] = rescale_qmap
        img[x_map-5:x_map+5, y_map-5:y_map+5, 2] = 1
        img = img   # To resize between 0 and 1 : to display with output probability
        return img

#### For Tensorboard #####
    def create_log(self):
        self.logger = tf.contrib.summary.create_file_writer(logdir='logs')

    @tfmpl.figure_tensor
    def draw_scatter(self, data):
        '''Draw scatter plots. One for each color.'''
        fig = tfmpl.create_figure()
        ax = fig.add_subplot(111)
        try:
            ax.imshow(data[:, :, :])
        except:
            ax.imshow(data[:, :, 0])
        fig.tight_layout()
        return fig

    @tfmpl.figure_tensor
    def draw_scatter_subplot(self, data1, data2):
        '''Draw scatter plots. One for each color.'''

        fig = tfmpl.create_figure()
        ax = fig.add_subplot(122)
        ax.imshow(data1[:, :, :])
        ax = fig.add_subplot(121)
        try:
            ax.imshow(data2[:, :, :])
        except:
            ax.imshow(data2[:, :, 0])
        fig.tight_layout()
        return fig

    def log_fig(self, name, data):
        with self.logger.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image(name, data)

    def log_img(self, name, data):
        with self.logger.as_default(), tf.contrib.summary.always_record_summaries():
            if type(data).__module__ == np.__name__:
                # If it is a numpy array
                im2 = tf.convert_to_tensor(data[0].reshape((1, data.shape[1], data.shape[2], data.shape[3])))
            else:
                # If it is an Eager Tensor
                im2 = tf.reshape(data[0], (1, data.shape[1], data.shape[2], data.shape[3]))
            im2 = tf.dtypes.cast(im2, tf.uint8)
            print('type im2', type(im2))
            tf.contrib.summary.image(name, im2)

    def log_scalar(self, name, data):
        with self.logger.as_default(), tf.contrib.summary.always_record_summaries():
            try:
                tf.contrib.summary.scalar(name, data)
            except:
                pass

    def log_generic(self, name, data):
        with self.logger.as_default(), tf.contrib.summary.always_record_summaries():
            try:
                tf.contrib.summary.generic(name, data)
            except:
                pass

if __name__=='__main__':
    Network = Trainer(savetosnapshot=False, load=False, snapshot_file='reference')
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1

    best_pix = [103, 125, 0, 40, 10]
    # best_pix = [83, 76, 0, 30, 10]         # x, y, angle, e, lp

    out = Network.forward(im)
    print(out.shape)
    viz = Network.prediction_viz(out, im)
    np.save('outputforFrancois.npy', viz)
    plt.imshow(viz)
    plt.show()

    ### Viz Demonstration ###
    viz_demo = True
    if viz_demo:
        print('demo')
        x, y, angle, e, lp = best_pix[0], best_pix[1], best_pix[2], best_pix[3], best_pix[4]
        rect = div.draw_rectangle(e, angle, x, y, lp)
        im_test = np.zeros((224, 224, 3))
        im_test[:, :, 1] = im[0, :, :, 0]
        demo = cv2.fillConvexPoly(im_test, rect, color=3)
        plt.imshow(demo)
        plt.show()

    Network.main(best_pix, im)
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    out = Network.forward(im)
    viz = Network.prediction_viz(out, im)

    plt.imshow(viz)
    plt.show()
