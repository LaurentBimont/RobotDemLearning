# -*- coding: utf-8 -*-

import model as mod
import tensorflow as tf

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
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=5e-4, momentum=0.9)
        self.action = RM.RewardManager()
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
        self.exp_rpl = ExperienceReplay(["depth_heightmap", "label", "loss"])
        # Frequency
        self.viz_frequency = 20
        self.saving_frequency = 50
        
        # Initiate logger
        self.create_log()

        if savetosnapshot:
            checkpoint_directory = "checkpoint_1"
            self.snapshot_file = os.path.join(checkpoint_directory, snapshot_file)
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.myModel,
                                                  optimizer_step=tf.train.get_or_create_global_step())
            if load:
                latest_snapshot_file = tf.train.latest_checkpoint(checkpoint_directory)
                status = self.checkpoint.restore(latest_snapshot_file)
                print('Pre-trained model snapshot loaded from: {}'.format(latest_snapshot_file))
            else:
                self.checkpoint.save(self.snapshot_file)
                print('Creating snapshot : {}'.format(self.snapshot_file))

        self.loss = func.partial(tf.losses.huber_loss)
        ######                     Demonstration                            ######
        self.best_idx = [125, 103]

    def custom_loss(self):
        '''
        As presented in 'Deep Q-learning from Demonstrations', the loss value is highly impacted by the
        :return: Loss Value
        '''
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * 0.001

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
        # expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
        # plt.subplot(1, 4, 4)
        # plt.imshow(label.numpy()[0, :, :, 0])
        label = self.reduced_label(label)

        new_lab, label_numpy = self.output_prob.numpy(), label.numpy()
        # plt.subplot(1, 4, 1)
        # plt.imshow(label_numpy[0, :, :, 0])
        # plt.subplot(1, 4, 2)
        # plt.imshow(new_lab[0, :, :, 0])
        new_lab[label_numpy == 1] = 1
        new_lab[label_numpy == -1] = 0
        # new_lab[label_numpy != 1] /= 2
        # plt.subplot(1, 4, 3)
        # plt.imshow(new_lab[0, :, :, 0])
        # plt.show()
        label = tf.convert_to_tensor(new_lab)
        # print(new_lab[0,:,:,:].shape, self.output_prob[0,:,:,:].shape)

        # plt.subplot(1, 2, 1)
        # plt.imshow(new_lab[0, :, :, 0], vmin=0, vmax=1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(self.output_prob[0, :, :, 0], vmin=0, vmax=1)
        # plt.show()
        # self.loss_value = self.loss(label, self.output_prob, reduction=tf.losses.Reduction.SUM)
        self.loss_value = self.loss(label, self.output_prob)
        # print('La valeur de perte est {}'.format(self.loss_value.numpy()))
        # Tensorboard ouputs

        if (tf.train.get_global_step() is not None) \
                and (tf.train.get_global_step().numpy() % self.viz_frequency == 0) \
                and (not noBackprop):
            print('Printing to Tensorboard')
            img_tensorboard = self.prediction_viz(3*self.output_prob, self.image)
            img_tensorboard_target = self.prediction_viz(label.numpy(), self.image)
            subplot_viz = self.draw_scatter_subplot(img_tensorboard, img_tensorboard_target)
            self.log_fig('subplot_viz', subplot_viz)
            self.log_img('input', self.image)
            self.log_scalar('loss value_dem', self.loss_value)
            output_prob_plt = self.draw_scatter(self.output_prob[0])
        # Saving a snapshot file
        if (self.savetosnapshot)\
                and (tf.train.get_global_step() is not None)\
                and (tf.train.get_global_step().numpy()%self.saving_frequency == 0)\
                and (not noBackprop):
            self.save_model()
        return self.loss_value

    def compute_labels(self, label_value, best_pix_ind, shape=(224,224,3), viz=False):
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
            plt.subplot(1, 3, 1)
            self.image = np.reshape(self.image, (self.image.shape[1], self.image.shape[2], 3))
            plt.imshow(self.image)
            plt.subplot(1, 3, 2)
            label_viz = np.reshape(label, (label.shape[0], label.shape[1]))
            plt.imshow(label_viz)
        return label

    def reduced_label(self, label):
        '''Reduce label Q-map to the output dimension of the network
        :param label: 224x224 label map
        :param label_weights:  224x224 label weights map
        :return: label and label_weights in output format
        '''

        label = tf.convert_to_tensor(label, np.float32)
        label = tf.image.resize_images(label, (self.width, self.height))
        label = tf.reshape(label[:, :, :, 0], (self.batch, self.width, self.height, 1))
        if self.classifier_boolean:
            label = label.numpy()
            label[label > 0.] = 1
            label = tf.convert_to_tensor(label, np.float32)
        return label

    def main_batches(self, im, label):
        self.future_reward = 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.myModel.trainable_variables)
            self.forward(im)
            self.compute_loss_dem(label)

            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())
            self.iteration = tf.train.get_global_step()

        self.exp_rpl.store([im, label, self.loss_value])

    def main_without_backprop(self, im, best_pix, batch_size=1, augmentation_factor=4, demo=True):
        # label = self.compute_labels(1, best_pix, shape=im.shape)
        label = best_pix
        im = resize(im, (224, 224, 3), anti_aliasing=True)
        label = resize(label, (224, 224, 3), anti_aliasing=True)
        # plt.subplot(1, 2, 1)
        # plt.imshow(im[:, :, 0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(label[:, :, 0])
        # plt.show()
        print('Data Augmentation')
        dataset = da.OnlineAugmentation().generate_batch(im, label, np.mean(im), viz=False, augmentation_factor=augmentation_factor)

        for batch in range(len(dataset['im'])//batch_size):
            batch_im, batch_label = self.random_batch(batch_size, dataset)
            self.forward(batch_im)
            self.compute_loss_dem(batch_label, noBackprop=True)
            self.exp_rpl.store([batch_im, batch_label, self.loss_value])
            if batch % 20 == 0:
                print('{}/{}'.format(batch, len(dataset['im'])//batch_size))

    def main_xpreplay(self, nb_epoch=2, batch_size=3):
        self.exp_rpl.generate_ranking()
        for epoch in range(nb_epoch):
            for batch in range(100):
                if batch % 20 == 0:
                    print('Epoch {}/{}, Batch {}/{}'.format(epoch + 1, nb_epoch, batch + 1, 100))
                batch_im, batch_lab = self.exp_rpl.replay(batch_size=batch_size)
                self.main_batches(batch_im, batch_lab)

    def random_batch(self, batch_size, dataset):
        im_o, label_o = dataset['im'], dataset['label']
        batch_tmp_im, batch_tmp_lab = [], []
        for i in range(batch_size):
            ind_tmp = np.random.randint(len(dataset['im']))
            batch_tmp_im.append(im_o[ind_tmp])
            batch_tmp_lab.append(label_o[ind_tmp])
        batch_im, batch_lab = tf.stack(batch_tmp_im), tf.stack(batch_tmp_lab)
        return batch_im, batch_lab

    def main(self, best_pix, im, epoch_size=1, batch_size=1, augmentation_factor=4):
        # label = self.compute_labels(1, best_pix, shape=im.shape)
        label = best_pix
        plt.subplot(1, 2, 1)
        plt.imshow(label)
        plt.subplot(1, 2, 2)
        plt.imshow(im)
        plt.show()
        im = resize(im, (224, 224, 3), anti_aliasing=True)
        label = resize(label, (224, 224, 3), anti_aliasing=True)
        dataset = da.OnlineAugmentation().generate_batch(im, label, np.mean(im), viz=False, augmentation_factor=augmentation_factor)
        for epoch in range(epoch_size):
            for batch in range(len(dataset['im']) // batch_size):
                print('Epoch {}/{}, Batch {}/{}'.format(epoch + 1, epoch_size, batch + 1,
                                                        len(dataset['im']) // batch_size))
                batch_im, batch_lab = self.random_batch(batch_size, dataset)
                print('LAAAA')
                # plt.subplot(1,2,1)
                # plt.imshow(batch_lab.numpy()[0,:,:,0])
                # plt.subplot(1, 2, 2)
                # plt.imshow(batch_im.numpy()[0, :, :, 0])
                # plt.show()
                self.main_batches(batch_im, batch_lab)

        # batch_im, batch_lab = self.exp_rpl.replay()
        # self.main_batches(batch_im, batch_lab)
        print('Finish XP replay')

#### Accessoires #######
    def vizualisation(self, img, idx):
        prediction = cv2.circle(img[0], (int(idx[1]), int(idx[0])), 7, (255, 255, 255), 2)
        plt.imshow(prediction)
        plt.show()

    def save_model(self):
        print("Saving model to {}".format(self.snapshot_file))
        self.checkpoint.save(self.snapshot_file)

    def prediction_viz(self, qmap, im):
        qmap1 = qmap
        qmap = qmap[0, :, :, :]
        qmap = tf.image.resize_images(qmap, (self.width, self.height))
        qmap = tf.image.resize_images(qmap, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        qmap = tf.reshape(qmap, (224, 224))

        rescale_qmap = qmap
        img = np.zeros((224, 224, 3))
        img[:, :, 0] = im[0, :, :, 0]
        img[:, :, 1] = rescale_qmap
        img = img / np.max(img)
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
    Network = Trainer(savetosnapshot=False, load=True, snapshot_file='reference')
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1

    best_pix = [103, 125, 0, 40, 10]
    best_pix = [83, 76, 0, 20, 10]         # x, y, angle, e, lp

    out = Network.forward(im)
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
        im_test[:, :, 1] = im[:, :, 0]
        demo = cv2.fillConvexPoly(im_test, rect, color=3)
        plt.imshow(demo)
        plt.show()

    Network.main(best_pix, im)

    #
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    out = Network.forward(im)
    viz = Network.prediction_viz(out, im)

    plt.imshow(viz)
    plt.show()
