import numpy as np
import pandas as pd
import tensorflow as tf 
class ExperienceReplay:
    def __init__(self,trainer,dataNames):
        self.header = dataNames
        self.replay_buffer = [] 
        self.trainer = trainer
    def store(self,data):
        if len(self.header) !=  len(data):
            raise ValueError("wrong number of stored data {} expected vs {}".format(len(self.header), len(data)))
        self.replay_buffer.append(data)

    def replay(self):
        if len(self.replay_buffer)>2:
            print("experience replay {} experience are stored in buffer ".format(len(self.replay_buffer)))
            data = pd.DataFrame(self.replay_buffer, columns=self.header)
            if len(data)==0:
                return

            # We sort the buffer by increasing loss 

            data = data.sort_values("loss")
            
            if len(data) > 1000:
                data=data.iloc[-1000:]

            pow_law_exp = 2

            rand_sample_idx = (np.random.power(pow_law_exp, 3)*(len(data)-1)).astype(np.int64)

            print("max experience loss : {}".format(data["loss"].iloc[-1]))

            # We sample three action in order to replay them. The more the loss the more the chance for an action to be taken
            data = data.iloc[rand_sample_idx]

            print("Nb data for experience replay : {}".format(data.shape))
            for idx, row in data.iterrows():
                heightmap, label, _ =row
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.trainer.myModel.trainable_variables)
                    self.trainer.forward(heightmap)
                    self.trainer.compute_loss_dem(label)

                    grad = tape.gradient(self.trainer.loss_value, self.trainer.myModel.trainable_variables)
                    self.trainer.optimizer.apply_gradients(zip(grad, self.trainer.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())
                    self.trainer.iteration = tf.train.get_global_step()

                    self.replay_buffer[idx][-1] = self.trainer.loss_value.numpy()
