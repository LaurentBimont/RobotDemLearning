import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
class ExperienceReplay:
    def __init__(self, dataNames):
        self.header = dataNames
        self.replay_buffer = []
        self.data = None

    def clean(self):
        self.replay_buffer = []
        print('Cleaning Buffer')

    def store(self, data, demo = False):
        if len(self.header) != len(data):
            raise ValueError("wrong number of stored data {} expected vs {}".format(len(self.header), len(data)))


        self.replay_buffer.append(data)

    def generate_ranking(self):
        print(len(self.replay_buffer))
        if len(self.replay_buffer) > 2:
            print("experience replay {} experience are stored in buffer ".format(len(self.replay_buffer)))
            self.data = pd.DataFrame(self.replay_buffer, columns=self.header)
            print(len(self.data))
            if len(self.data) == 0:
                return
            # We sort the buffer by increasing loss
            self.data = self.data.sort_values("loss")
            if len(self.data) > 1000:
                self.data = self.data.iloc[-1000:]
            print(self.data.shape)

    def replay(self, batch_size=3):
        if self.data is not None:
            if len(self.data) != 0:
                pow_law_exp = 2
                rand_sample_idx = (np.random.power(pow_law_exp, batch_size)*(len(self.data)-1)).astype(np.int64)
                # print("max experience loss : {}".format(data["loss"].iloc[-1]))
                # We sample batch_size number of actions in order to replay them. The more the loss the more the chance for an action to be taken
                subsample = self.data.iloc[rand_sample_idx]
                # plt.subplot(1, 2, 1)
                # plt.imshow(subsample['label'].values[0].numpy()[0, :, :, 0])
                # plt.subplot(1, 2, 2)
                # plt.imshow(subsample['depth_heightmap'].values[0].numpy()[0, :, :, 0])
                # plt.show()

                # batch_img, batch_lab = tf.stack(data['depth_heightmap'].values),\
                #                        tf.stack(data['label'].values)

                batch_img, batch_lab = [], []
                # print(self.data.shape[0])
                for i in range(subsample.shape[0]):
                    batch_img.append(subsample['depth_heightmap'].values[i][0])
                    batch_lab.append(subsample['label'].values[i][0])
                # batch_img, batch_lab = data['depth_heightmap'].values[1], data['label'].values[1]
                batch_img, batch_lab = tf.stack(batch_img), tf.stack(batch_lab)
                # print(type(batch_img))
                # print(batch_img.shape)
                return batch_img, batch_lab

            return None, None
            ### Useless ###
            # print("Nb data for experience replay : {}".format(data.shape))
            # for idx, row in data.iterrows():
            #     heightmap, label, _ =row
            #     with tf.GradientTape(watch_accessed_variables=False) as tape:
            #         tape.watch(self.trainer.myModel.trainable_variables)
            #         self.trainer.forward(heightmap)
            #         self.trainer.compute_loss_dem(label)
            #         grad = tape.gradient(self.trainer.loss_value, self.trainer.myModel.trainable_variables)
            #         self.trainer.optimizer.apply_gradients(zip(grad, self.trainer.myModel.trainable_variables),
            #                                global_step=tf.train.get_or_create_global_step())
            #         self.trainer.iteration = tf.train.get_global_step()
            #
            #         self.replay_buffer[idx][-1] = self.trainer.loss_value.numpy()
