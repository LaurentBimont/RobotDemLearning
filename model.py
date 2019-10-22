import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import divers as div

if __name__=="__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.enable_eager_execution(config)

class DensenetFeatModel(tf.keras.Model):
    def __init__(self):
        '''
        Dense net est entraîné sur des images 224x224, si l'image d'entrée est plus grande le réseau va appliquer
        fois Densenet sur des sous-régions, jusqu'à obtenir l'image complète
        '''

        super(DensenetFeatModel, self).__init__()
        baseModel = tf.keras.applications.densenet.DenseNet121(weights='imagenet')

        self.model = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.get_layer(
            "conv5_block16_concat").output)
        for layer in self.model.layers:
            layer.trainable = False

    def call(self, inputs):
        # inputs = tf.transpose(inputs,(0,3,2,1))
        output = self.model(inputs)
        return output


class GraspNetTest(BaseDeepModel):
    def __init__(self):
        super(GraspNetTest, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167
        # self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")
        dropout_rate = 0.4
        ### 1 ere couche ###
        self.conv0 = tf.keras.layers.Convolution2D(128, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv0", trainable=True)
        self.drop0 = tf.keras.layers.Dropout(dropout_rate)
        ### 2 ieme couche ###
        self.upconv0 = tf.keras.layers.UpSampling2D((2, 2))
        ### 3 ieme couche ###
        self.bn1 = tf.keras.layers.BatchNormalization(name="grasp-b1")
        self.conv1 = tf.keras.layers.Convolution2D(256, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1", trainable=True)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        ### Classification couche ###
        self.bn2 = tf.keras.layers.BatchNormalization(name="grasp-b2")
        self.conv2 = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.tanh,
                                                   use_bias=False, padding='same', name="grasp-conv2", trainable=True)

    def call(self, inputs, bufferize=False, step_id=-1):
        # print('Entrée du réseau seondaire', inputs.shape)
        # x = self.bn0(inputs)
        ### 1 ere couche ###
        x = self.conv0(inputs)
        x = self.drop0(x)
        ### 2 ieme couche ###
        x = self.upconv0(x)
        ### 3 ieme couche ###
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.drop1(x)
        ### Classification couche ###
        x = self.bn2(x)
        x = self.conv2(x)
        return x


class Reinforcement(tf.keras.Model):
    def __init__(self):
        super(Reinforcement, self).__init__()
        self.Dense = DensenetFeatModel()
        self.QGraspTest = GraspNetTest()
        # Initialize variables
        self.in_height, self.in_width = 0, 0
        self.scale_factor = 2.0
        self.padding_width = 0
        self.target_height = 0
        self.target_width = 0

    def call(self, input):
        x = self.QGraspTest(self.Dense(input))
        return x

if __name__ == "__main__":
    im = np.ndarray((1, 224, 224, 3), np.float32)
    Densenet = Reinforcement()
    print(Densenet(im).shape)
