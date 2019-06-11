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
        self.classifier_boolean = True
        self.savetosnapshot = savetosnapshot
        # self.create_log()

        # Frequency
        self.viz_frequency = 100
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

        ###### A changer par une fonction adaptée au demonstration learning ######
        # self.loss = func.partial(tf.losses.huber_loss)                  # Huber loss
        # self.loss = func.partial(tf.losses.sigmoid_cross_entropy)
        # self.loss = func.partial(tf.losses.mean_pairwise_squared_error)
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
        print('Output : ', self.width, self.height)
        # Return Q-map
        return self.output_prob

    def backpropagation(self, gradient):
        self.optimizer.apply_gradients(zip(gradient, self.myModel.trainable_variables),
                                       global_step=tf.train.get_or_create_global_step())
        self.iteration = tf.train.get_global_step()

    def compute_loss(self):
        # A changer pour pouvoir un mode démonstration et un mode renforcement
        expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
        label224, label_weights224 = self.compute_labels(expected_reward, self.best_idx)
        label, label_weights = self.reduced_label(label224, label_weights224)
        self.output_prob = tf.reshape(self.output_prob, (self.width, self.height, 1))
        self.loss_value = self.loss(label, self.output_prob, label_weights)
        return self.loss_value

    def compute_loss_dem(self, label, label_w, viz=False):
        expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
        if viz:
            plt.subplot()
            plt.imshow(label[0, :, :, :])
            plt.show()

        label, label_weights = self.reduced_label(label, label_w)
        #self.output_prob = tf.reshape(self.output_prob, (self.batch, self.width, self.height, 1))


        self.loss_value = self.loss(label, self.output_prob)

        # Tensorboard ouputs
        if (tf.train.get_global_step() is not None) and (tf.train.get_global_step().numpy()%self.viz_frequency == 0):

            ##### Visualisation sortie de réseaux
            # plt.subplot(1, 2, 1)
            # plt.imshow(label[0, :, :, 0], vmin=0, vmax=1)
            # plt.subplot(1, 2, 2)
            # plt.imshow(self.output_prob[0, :, :, 0], vmin=0, vmax=1)
            # plt.show()

            img_tensorboard = self.prediction_viz(3*self.output_prob, self.image)
            img_tensorboard_target = self.prediction_viz(label.numpy(), self.image)
            #img_tensorboard_target_plot = self.draw_scatter(img_tensorboard_target)
            #self.log_fig('viz', img_tensorboard_plot)
            #self.log_fig('vizage', img_tensorboard_target_plot)

            # plt.subplot(1, 2, 1)
            # plt.imshow(img_tensorboard)
            # plt.subplot(1, 2, 2)
            # plt.imshow(img_tensorboard_target)
            # plt.show()

            subplot_viz = self.draw_scatter_subplot(img_tensorboard, img_tensorboard_target)
            self.log_fig('subplot_viz', subplot_viz)
            self.log_img('input', self.image)
            # inutile pour le moment
            # tensorboard_output_prob = self.output_viz(self.output_prob)

            self.log_scalar('loss value_dem', self.loss_value)

            output_prob_plt = self.draw_scatter(self.output_prob[0])
            # self.log_fig('output_prob_plt', output_prob_plt)

            # Save a snapshot of the model

            # np.save('label{}'.format(self.num), label.numpy())
            # np.save('output_prob{}'.format(self.num), self.output_prob.numpy())
            # np.save('computed_loss{}'.format(self.num), np.array([self.loss_value]))
            # self.num += 1

        # Saving a snapshot file
        if (self.savetosnapshot) and (tf.train.get_global_step() is not None) and (tf.train.get_global_step().numpy()%self.saving_frequency == 0):
            self.save_model()

        return self.loss_value

    def compute_labels(self, label_value, best_pix_ind, viz=False):
        '''Create the targeted Q-map
        :param label_value: Reward of the action
        :param best_pix_ind: (Rectangle Parameters : x(colonne), y(ligne), angle(en degré), ecartement(en pixel)) Pixel where to perform the action
        :return: label : an 224x224 array where best pix is at future reward value
                 label_weights : a 224x224 where best pix is at one
        '''
        # Compute labels
        x, y, angle, e, lp = best_pix_ind[0], best_pix_ind[1], best_pix_ind[2], best_pix_ind[3], best_pix_ind[4]
        rect = div.draw_rectangle(e, angle, x, y, lp)

        label = np.zeros((224, 224, 3), dtype=np.float32)
        cv2.fillConvexPoly(label, rect, color=1)
        label *= label_value

        label_weights = np.ones((224, 224, 3), dtype=np.float32)
        if viz:
            plt.subplot(1, 3, 1)
            self.image = np.reshape(self.image, (self.image.shape[1], self.image.shape[2], 3))
            plt.imshow(self.image)
            plt.subplot(1, 3, 2)
            label_viz = np.reshape(label, (label.shape[0], label.shape[1]))
            plt.imshow(label_viz)
        return label, label_weights

    def reduced_label(self, label, label_weights, viz=False):
        '''Reduce label Q-map to the output dimension of the network
        :param label: 224x224 label map
        :param label_weights:  224x224 label weights map
        :return: label and label_weights in output format
        '''

        # label, label_weights = label[:, :, 0], label_weights[:, :, 0]
        if viz:
            plt.subplot(1, 2, 1)
            plt.imshow(label[0, :, :, 0])

        label, label_weights = tf.convert_to_tensor(label, np.float32),\
                               tf.convert_to_tensor(label_weights, np.float32)
        label, label_weights = tf.image.resize_images(label, (self.width, self.height)),\
                               tf.image.resize_images(label_weights, (self.width, self.height))
        label, label_weights = tf.reshape(label[:, :, :, 0], (self.batch, self.width, self.height, 1)), \
                               tf.reshape(label_weights[:, :, :, 0], (self.batch, self.width, self.height, 1))

        if self.classifier_boolean:
            label = label.numpy()
            label[label > 0.] = 1
            label = tf.convert_to_tensor(label, np.float32)
        if viz:
            plt.subplot(1, 2, 2)
            plt.imshow(label.numpy()[0, :, :, 0])
            plt.show()
        return label, label_weights

    def output_viz(self, output_prob):
        output_viz = np.clip(output_prob, 0, 1)
        output_viz = cv2.applyColorMap((output_viz*255).astype(np.uint8), cv2.COLORMAP_JET)
        output_viz = cv2.cvtColor(output_viz, cv2.COLOR_BGR2RGB)
        return np.array([output_viz])

    def vizualisation(self, img, idx):
        prediction = cv2.circle(img[0], (int(idx[1]), int(idx[0])), 7, (255, 255, 255), 2)
        plt.imshow(prediction)
        plt.show()

    def main(self, input):
        self.future_reward = 1
        with tf.GradientTape() as tape:
            self.forward(input)
            self.compute_loss()
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())

    def main_augmentation(self, dataset):
        ima, val, val_w = dataset['im'], dataset['label'], dataset['label_weights']
        self.future_reward = 1
        for j in range(len(ima)):
            if j % 10 == 0:
                print('Iteration {}/{}'.format(j, len(ima)))
            with tf.GradientTape() as tape:
                self.forward(tf.reshape(ima[j], (1, 224, 224, 3)))
                self.compute_loss_dem(val[j], val_w[j])
                grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                               global_step=tf.train.get_or_create_global_step())

    def save_model(self):
        print("Saving model to {}".format(self.snapshot_file))
        self.checkpoint.save(self.snapshot_file)

    def main_batches(self, im, label, label_weights, viz=False):
        self.future_reward = 1
        with tf.GradientTape() as tape:
            self.forward(im)
            if viz:
                plt.imshow(label[0])
                plt.show()
            self.compute_loss_dem(label, label_weights, viz=False)
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())
            self.iteration = tf.train.get_global_step()

    def create_log(self):
        self.logger = tf.contrib.summary.create_file_writer(logdir='logs')

    def prediction_viz(self, qmap, im):
        qmap1 = qmap
        qmap = qmap[0, :, :, :]
        qmap = tf.image.resize_images(qmap, (self.width, self.height))
        qmap = tf.image.resize_images(qmap, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        qmap = tf.reshape(qmap, (224, 224))

        # if np.max(qmap.numpy()) - np.min(qmap.numpy()) == 0.0:
        #     print(im.shape)
        #     print(qmap1.shape)
        #     plt.subplot(2,3,1)
        #     plt.imshow(im[0, :, :, 0])
        #     plt.subplot(2, 3, 2)
        #     plt.imshow(im[1, :, :, 0])
        #     plt.subplot(2, 3, 3)
        #     plt.imshow(im[2, :, :, 0])
        #     plt.subplot(2, 3, 4)
        #     plt.imshow(qmap1[0, :, :, 0])
        #     plt.subplot(2, 3, 5)
        #     plt.imshow(qmap1[1, :, :, 0])
        #     plt.subplot(2, 3, 6)
        #     plt.imshow(qmap1[2, :, :, 0])
        #     plt.show()

        # rescale_qmap = (qmap.numpy() - np.min(qmap.numpy())) / (
        #         np.max(qmap.numpy()) - np.min(qmap.numpy()))

        rescale_qmap = qmap
        img = np.zeros((224, 224, 3))
        img[:, :, 0] = im[0, :, :, 0]
        img[:, :, 1] = rescale_qmap
        img = img / np.max(img)
        return img

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
    Network = Trainer(savetosnapshot=True, snapshot_file='reference')
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1

    best_pix = [103, 125, 0, 40, 10]
    best_pix = [83, 76, 0, 20, 10]         # x, y, angle, e, lp

    ### Viz Demonstration ###
    viz_demo = True
    if viz_demo:
        x, y, angle, e, lp = best_pix[0], best_pix[1], best_pix[2], best_pix[3], best_pix[4]
        rect = div.draw_rectangle(e, angle, x, y, lp)
        im_test = np.zeros((224, 224, 3))
        im_test[:, :, 1] = im[:, :, :, 0]
        demo = cv2.fillConvexPoly(im_test, rect, color=3)
        plt.imshow(demo)
        plt.show()
    # Test de Tensorboard

    # A REMETTRE SI CA MERDE
    # Network.create_log()
    previous_qmap = Network.forward(im)

    label, label_weights = Network.compute_labels(1, best_pix)
    dataset = da.OnlineAugmentation().generate_batch(im, label, label_weights, viz=False, augmentation_factor=6)
    im_o, label_o, label_wo = dataset['im'], dataset['label'], dataset['label_weights']
    epoch_size = 2
    batch_size = 1
    for epoch in range(epoch_size):
        for batch in range(len(dataset['im']) // batch_size):
            print('Epoch {}/{}, Batch {}/{}'.format(epoch + 1, epoch_size, batch + 1,
                                                    len(dataset['im']) // batch_size))
            batch_tmp_im, batch_tmp_lab, batch_tmp_weights = [], [], []
            for i in range(batch_size):
                ind_tmp = np.random.randint(len(dataset['im']))
                batch_tmp_im.append(im_o[ind_tmp])
                batch_tmp_lab.append(label_o[ind_tmp])
                batch_tmp_weights.append(label_wo[ind_tmp])

            batch_im, batch_lab, batch_weights = tf.stack(batch_tmp_im), tf.stack(batch_tmp_lab), tf.stack(
                batch_tmp_weights)

            Network.main_batches(batch_im, batch_lab, batch_weights)

    trained_qmap = Network.forward(im)
    ntrained_qmap = trained_qmap.numpy()

    # Creation of a rotated view
    im2 = sc.ndimage.rotate(im[0, :, :, :], 90)
    im2.reshape(1, im2.shape[0], im2.shape[1], im2.shape[2])
    im2 = np.array([im2])

    # Resizes images
    new_qmap = Network.forward(im2)

    # New Q-map vizualisation
    plt.subplot(1, 3, 3)
    img = Network.prediction_viz(3*new_qmap, im2)
    plt.imshow(img)

    # First Q-map vizualisation
    plt.subplot(1, 3, 1)
    img = Network.prediction_viz(previous_qmap, im)
    plt.imshow(img)

    # First Q-map vizualisation after training
    plt.subplot(1, 3, 2)
    img = Network.prediction_viz(3*trained_qmap, im)
    plt.imshow(img)

    plt.show()
