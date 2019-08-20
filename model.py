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
            layer.trainable = True

    def call(self, inputs):
        # inputs = tf.transpose(inputs,(0,3,2,1))
        output = self.model(inputs)
        return output

class VGGFeatModel(tf.keras.Model):
    def __init__(self):
        '''
        Dense net est entraîné sur des images 224x224, si l'image d'entrée est plus grande le réseau va appliquer
        fois Densenet sur des sous-régions, jusqu'à obtenir l'image complète
        '''
        super(VGGFeatModel, self).__init__()
        baseModel = tf.keras.applications.VGG19(weights='imagenet')
        self.model = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.get_layer("block5_pool").output)

    def call(self, inputs):
        # inputs = tf.transpose(inputs,(0,3,2,1))
        output = self.model(inputs)
        return output

class BaseDeepModel(tf.keras.Model):
    def __init__(self):
        super(BaseDeepModel, self).__init__()
        pass

class PixelNet(BaseDeepModel):
    def __init__(self):
        super(PixelNet, self).__init__()
        self.flat = tf.keras.layers.Flatten()
        self.D1 = tf.keras.layers.Dense(512, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.D2 = tf.keras.layers.Dense(1024, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.D3 = tf.keras.layers.Dense(2048, activation='relu')
        self.D4 = tf.keras.layers.Dense(441, activation='sigmoid')

    def call(self, inputs, bufferize=False, step_id=-1):
        x = self.flat(inputs)
        x = self.D1(x)
        x = self.D2(x)
        x = self.D3(x)
        x = self.D4(x)
        x = tf.reshape(x, (1, 21, 21, 1))
        return x

class OnlyConvNet(BaseDeepModel):
    def __init__(self):
        super(OnlyConvNet, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167
        self.conv0 = tf.keras.layers.Convolution2D(32, kernel_size=(9, 9), strides=3, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv0",
                                                   trainable=True)
        self.conv1 = tf.keras.layers.Convolution2D(16, kernel_size=(5, 5), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1",
                                                   trainable=True)
        self.conv2 = tf.keras.layers.Convolution2D(8, kernel_size=(3, 3), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv2",
                                                   trainable=True)
        self.tconv0 = tf.keras.layers.Conv2DTranspose(8, kernel_size=(3, 3), strides=2, activation=tf.nn.relu,
                                                      use_bias=True, padding='same', name="tgrasp_conv0",
                                                      trainable=True)
        self.tconv1 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=2, activation=tf.nn.relu,
                                                      use_bias=True, padding='same', name="tgrasp_conv0",
                                                      trainable=True)
        self.tconv2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(9, 9), strides=3, activation=tf.nn.relu,
                                                      use_bias=True, padding='same', name="tgrasp_conv0",
                                                      trainable=True)
        self.outputconv = tf.keras.layers.Convolution2D(1, kernel_size=(2, 2), strides=2, activation=tf.nn.sigmoid,
                                                   use_bias=True, padding='same', name="grasp-conv2",
                                                   trainable=True)

    def call(self, inputs, bufferize=False, step_id=-1):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.outputconv(x)
        return(x)

class PrincetonNet(BaseDeepModel):
    def __init__(self):
        super(OnlyConvNet, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167
        self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")

        self.conv0 = tf.keras.layers.Convolution2D(32, kernel_size=(9, 9), strides=3, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv0",
                                                   trainable=True)
        self.conv1 = tf.keras.layers.Convolution2D(16, kernel_size=(5, 5), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1",
                                                   trainable=True)
        self.conv2 = tf.keras.layers.Convolution2D(8, kernel_size=(3, 3), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv2",
                                                   trainable=True)


    def call(self, inputs, bufferize=False, step_id=-1):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.outputconv(x)
        return(x)

class GraspNetTest(BaseDeepModel):
    def __init__(self):
        super(GraspNetTest, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167

        self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")
        self.conv0 = tf.keras.layers.Convolution2D(128, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv0", trainable=True)

        self.bn1 = tf.keras.layers.BatchNormalization(name="grasp-b1")

        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.conv1 = tf.keras.layers.Convolution2D(256, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1", trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(name="grasp-b2")

        self.drop2 = tf.keras.layers.Dropout(0.1)
        self.conv2 = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.tanh,
                                                   use_bias=False, padding='same', name="grasp-conv2", trainable=True)
        self.bn3 = tf.keras.layers.BatchNormalization(name="grasp-b2")

        # self.tconv0 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=3, activation=tf.nn.relu,
        #                                               use_bias=True, padding='same', name="tgrasp_conv0",
        #                                               trainable=True)

    def call(self, inputs, bufferize=False, step_id=-1):
        # print('Entrée du réseau seondaire', inputs.shape)
        # x = self.bn0(inputs)
        x = self.conv0(inputs)
        # x = self.drop1(x)
        x = self.bn1(x)
        x = self.conv1(x)
        # x = self.drop2(x)
        x = self.bn2(x)
        x = self.conv2(x)
        # x = self.bn3(x)
        # x = tf.multiply(3, x)
        # Rescaling between 0 and 1
        # x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
        # x = tf.sigmoid(self.bn2(x))
        # x = self.conv3(x)
        # x = tf.math.add(tf.math.add(x[:, :, :, 0], x[:, :, :, 1]), x[:, :, :, 2])
        # x = x[:, :, :, 0]
        # x = tf.math.divide(x, tf.constant(3, dtype=tf.float32))
        # x = tf.reshape(x, (*x.shape, 1))
        # for i in range(x.shape[0]):
        #     x[i, :, :, :] = tf.div(tf.subtract(x[i, :, :, :], tf.reduce_min(x[i, :, :, :])),
        #                            tf.subtract(tf.reduce_max(x[i, :, :, :]), tf.reduce_min(x[i, :, :, :])))
        # x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
        return x

class GraspNet(BaseDeepModel):
    def __init__(self):
        super(GraspNet, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167

        self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")
        self.conv0 = tf.keras.layers.Convolution2D(128, kernel_size=1, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv0", trainable=True)
        self.bn1 = tf.keras.layers.BatchNormalization(name="grasp-b1")

        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.conv1 = tf.keras.layers.Convolution2D(256, kernel_size=1, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1", trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(name="grasp-b2")

        self.drop2 = tf.keras.layers.Dropout(0.2)
        # self.conv2 = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.sigmoid,
        #                                            use_bias=False, padding='same', name="grasp-conv2", trainable=True)
        # self.bn3 = tf.keras.layers.BatchNormalization(name="grasp-b2")

        self.position_conv= tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.sigmoid,
                                                   use_bias=False, padding='same', name="pos-conv", trainable=True)
        self.position_bn = tf.keras.layers.BatchNormalization(name="position-bn")

        self.cos_conv = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.sigmoid,
                                                         use_bias=False, padding='same', name="cos-conv", trainable=True)

        self.cos_bn = tf.keras.layers.BatchNormalization(name="cos-bn")
        self.sin_conv = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.sigmoid,
                                                 use_bias=False, padding='same', name="sin-conv", trainable=True)
        self.sin_bn = tf.keras.layers.BatchNormalization(name="sin-bn")

        self.width_conv = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.sigmoid,
                                                    use_bias=False, padding='same', name="width-conv", trainable=True)
        self.width_bn = tf.keras.layers.BatchNormalization(name="width-bn")

        # self.tconv0 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=3, activation=tf.nn.relu,
        #                                               use_bias=True, padding='same', name="tgrasp_conv0",
        #                                               trainable=True)
        
    def call(self, inputs, bufferize=False, step_id=-1):
        # print('Entrée du réseau seondaire', inputs.shape)
        #x = self.bn0(inputs)
        x = self.conv0(inputs)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.drop2(x)
        # x = self.tconv0(x)
        # x = self.conv2(x)
        

        pos = self.position_bn(self.position_conv(x)) 
        cos = self.cos_bn(self.cos_conv(x)) 
        sin = self.sin_bn(self.sin_conv(x)) 
        width = self.width_bn(self.width_conv(x))
        return pos, cos, sin, width
    # x = self.bn3(x)
    # x = tf.multiply(3, x)
    # Rescaling between 0 and 1
    # x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
    # x = tf.sigmoid(self.bn2(x))
            # x = self.conv3(x)
            # x = tf.math.add(tf.math.add(x[:, :, :, 0], x[:, :, :, 1]), x[:, :, :, 2])
            # x = x[:, :, :, 0]
            # x = tf.math.divide(x, tf.constant(3, dtype=tf.float32))
            # x = tf.reshape(x, (*x.shape, 1))
            # for i in range(x.shape[0]):
            #     x[i, :, :, :] = tf.div(tf.subtract(x[i, :, :, :], tf.reduce_min(x[i, :, :, :])),
            #                            tf.subtract(tf.reduce_max(x[i, :, :, :]), tf.reduce_min(x[i, :, :, :])))
            # x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
            # return x
        
class Reinforcement(tf.keras.Model):
    def __init__(self):
        super(Reinforcement, self).__init__()
        self.Dense = DensenetFeatModel()
        # self.VGG = VGGFeatModel()
        self.QGrasp = GraspNet()
        self.QGraspTest = GraspNetTest()
        self.PixNet = PixelNet()
        self.OnlyConvNet = OnlyConvNet()
        # self.my_trainable_variables = self.QGrasp.trainable_variables

        # Initialize variables
        self.in_height, self.in_width = 0, 0
        self.scale_factor = 2.0
        self.padding_width = 0
        self.target_height = 0
        self.target_width = 0

    def call(self, input):
        # x = self.QGrasp(self.VGG(input))
        # pos, cos, sin, width = self.QGrasp(input)
        # x = self.QGrasp(self.Dense(input))
        # x = self.PixNet(self.Dense(input))
        # x = self.QGrasp(input)
        # x = self.OnlyConvNet(input)
        x = self.QGraspTest(self.Dense(input))
        # return pos, cos, sin, width
        return x

if __name__ == "__main__":
    im = np.ndarray((3, 224, 224, 3), np.float32)
    Densenet = Reinforcement()
    print(Densenet(im))
