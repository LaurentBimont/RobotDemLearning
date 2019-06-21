import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
class ExperienceReplay:
    def __init__(self, trainer, dataNames):
        self.header = dataNames
        self.replay_buffer = [] 
        self.trainer = trainer

    def clean(self):
        self.replay_buffer = []

    def store(self, data):
        if len(self.header) != len(data):
            raise ValueError("wrong number of stored data {} expected vs {}".format(len(self.header), len(data)))
        self.replay_buffer.append(data)

    def replay(self, batch_size=3):
        if len(self.replay_buffer) > 2:
            print("experience replay {} experience are stored in buffer ".format(len(self.replay_buffer)))
            data = pd.DataFrame(self.replay_buffer, columns=self.header)
            if len(data) == 0:
                return
            # We sort the buffer by increasing loss
            data = data.sort_values("loss")
            if len(data) > 1000:
                data = data.iloc[-1000:]
            pow_law_exp = 2
            rand_sample_idx = (np.random.power(pow_law_exp, batch_size)*(len(data)-1)).astype(np.int64)
            print("max experience loss : {}".format(data["loss"].iloc[-1]))
            # We sample batch_size number of actions in order to replay them. The more the loss the more the chance for an action to be taken
            data = data.iloc[rand_sample_idx]
            print('la')
            print(data['depth_heightmap'])
            print('aaaaaaaaaa')
            print(data['depth_heightmap'].values.shape)
            print(data['depth_heightmap'].values)
            plt.imshow(data['depth_heightmap'].values.numpy[0,:,:,0])
            plt.show()
            print('aaaaaaaaaa')
            print(tf.stack(data['depth_heightmap'].values))
            print('aaaaaaaaaa')
            batch_img, batch_lab = tf.stack(data['depth_heightmap'].values),\
                                   tf.stack(data['label'].values)
            print('la')
            return batch_img, batch_lab
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
