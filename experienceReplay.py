import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class ExperienceReplay:
    def __init__(self, dataNames):
        self.header = dataNames
        self.replay_buffer = []
        self.replay_buffer_demo = []
        self.data =[] 
        self.data_demo = [] 
    def clean(self):
        self.replay_buffer = []
        print('Cleaning Buffer')

    def store(self, data,demo=False):
        if len(self.header) != len(data):
            raise ValueError("wrong number of stored data {} expected vs {}".format(len(self.header), len(data)))
        if demo:
            self.replay_buffer_demo.append(data) 
        else:
            self.replay_buffer.append(data)

    def generate_ranking(self):
        if len(self.replay_buffer_demo)>0:
            self.data_demo = pd.DataFrame(self.replay_buffer_demo, columns=self.header)
            self.data_demo = self.data_demo.sort_values("loss")
        if len(self.replay_buffer) > 0:
            print("experience replay {} experience are stored in buffer ".format(len(self.replay_buffer)))
            self.data = pd.DataFrame(self.replay_buffer, columns=self.header)
            if len(self.data) == 0:
                return
            # We sort the buffer by increasing loss
            self.data = self.data.sort_values("loss")

            if len(self.data) > 1000:
                self.data = self.data.iloc[-1000:]

    def replay(self, batch_size=3):
        pow_law_exp = 2
        rand_sample_idx_demo = (np.random.power(pow_law_exp, batch_size)*(len(self.data_demo)-1)).astype(np.int64)

        subsample_demo = self.data_demo.iloc[rand_sample_idx_demo]
        subsample = [] 
        if len(self.data) != 0 :
            if len(self.data) != 0:
                pow_law_exp = 2
                rand_sample_idx = (np.random.power(pow_law_exp, batch_size)*(len(self.data)-1)).astype(np.int64)
                subsample = self.data.iloc[rand_sample_idx]

        batch_img, batch_lab = [], []

        if len(subsample) != 0:
            subsample = subsample[int(0.33*batch_size):]
            subsample = subsample + subsample_demo[:batch_size - len(subsample)]
        else:
            subsample = subsample_demo
        for i in range(subsample.shape[0]):
            batch_img.append(subsample['depth_heightmap'].values[i][0])

            batch_lab.append(subsample['label'].values[i][0])
        batch_img, batch_lab = tf.stack(batch_img), tf.stack(batch_lab)

        return batch_img, batch_lab
