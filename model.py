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
        # self.bn3 = tf.keras.layers.BatchNormalization(name="grasp-b2")

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

class GraspUNet(BaseDeepModel):
    def __init__(self):
        super(GraspUNet, self).__init__()
        nb_filter = 16
        # Contracting path
        ## First level
        self.c1_a = tf.keras.layers.Convolution2D(nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1", trainable=True)
        self.c1_b = tf.keras.layers.Convolution2D(nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv2", trainable=True)
        self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop1 = tf.keras.layers.Dropout(0.2)
        ## Second level
        self.c2_a = tf.keras.layers.Convolution2D(2*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv3", trainable=True)
        self.c2_b = tf.keras.layers.Convolution2D(2*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv4", trainable=True)
        self.mp2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop2 = tf.keras.layers.Dropout(0.2)
        ## Third level
        self.c3_a = tf.keras.layers.Convolution2D(4*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv5", trainable=True)
        self.c3_b = tf.keras.layers.Convolution2D(4*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv6", trainable=True)
        self.mp3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop3 = tf.keras.layers.Dropout(0.2)
        ## Fourth level
        self.c4_a = tf.keras.layers.Convolution2D(8*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv7", trainable=True)
        self.c4_b = tf.keras.layers.Convolution2D(8*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv8", trainable=True)
        self.mp4 = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop4 = tf.keras.layers.Dropout(0.2)

        # Truc tout seul
        self.c5_a = tf.keras.layers.Convolution2D(16*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv9", trainable=True)
        self.c5_b = tf.keras.layers.Convolution2D(16*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                use_bias=True, padding='same', name="grasp-conv10", trainable=True)

        # Expansive path
        # Sixth Level
        self.tc6 = tf.keras.layers.Conv2DTranspose(8*nb_filter, kernel_size=(2, 2), strides=2, activation=tf.nn.relu,
                                                      use_bias=True, padding='same', name="tgrasp_conv0",
                                                      trainable=True)
        self.drop6 = tf.keras.layers.Dropout(0.2)
        self.c6_a = tf.keras.layers.Convolution2D(8*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv11", trainable=True)
        self.c6_b = tf.keras.layers.Convolution2D(8*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv12", trainable=True)

        #Seventh level
        self.tc7 = tf.keras.layers.Conv2DTranspose(4*nb_filter, kernel_size=(2, 2), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="tgrasp_conv1",
                                                   trainable=True)
        self.drop7 = tf.keras.layers.Dropout(0.2)
        self.c7_a = tf.keras.layers.Convolution2D(4*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv13", trainable=True)
        self.c7_b = tf.keras.layers.Convolution2D(4*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv14", trainable=True)

        # Eigth level
        self.tc8 = tf.keras.layers.Conv2DTranspose(2*nb_filter, kernel_size=(2, 2), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="tgrasp_conv2",
                                                   trainable=True)
        self.drop8 = tf.keras.layers.Dropout(0.2)
        self.c8_a = tf.keras.layers.Convolution2D(2 * nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv15", trainable=True)
        self.c8_b = tf.keras.layers.Convolution2D(2*nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv16", trainable=True)

        # Ninth Level
        self.tc9 = tf.keras.layers.Conv2DTranspose(nb_filter, kernel_size=(2, 2), strides=2, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="tgrasp_conv3",
                                                   trainable=True)
        self.drop9 = tf.keras.layers.Dropout(0.2)
        self.c9_a = tf.keras.layers.Convolution2D(nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv17", trainable=True)
        self.c9_b = tf.keras.layers.Convolution2D(nb_filter, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                  use_bias=True, padding='same', name="grasp-conv18", trainable=True)

        # Output
        self.outconv = tf.keras.layers.Convolution2D(1, kernel_size=1, trainable=True, activation=tf.nn.tanh,
                                                     use_bias=True)


    def call(self, inputs, bufferize=False, step_id=-1):
        # Contracting Path
        c1 = self.c1_a(inputs)
        c1 = self.c1_b(inputs)
        p1 = self.mp1(c1)
        p1 = self.drop1(p1)

        c2 = self.c2_a(p1)
        c2 = self.c2_b(c2)
        p2 = self.mp2(c2)
        p2 = self.drop2(p2)

        c3 = self.c3_a(p2)
        c3 = self.c3_b(c3)
        p3 = self.mp3(c3)
        p3 = self.drop3(p3)

        c4 = self.c4_a(p3)
        c4 = self.c4_b(c4)
        p4 = self.mp4(c4)
        p4 = self.drop4(p4)

        c5 = self.c5_a(p4)
        c5 = self.c5_b(c5)

        # Expansive Path
        u6 = self.tc6(c5)
        u6 = tf.concat([u6, c4], axis=3)
        u6 = self.drop6(u6)
        c6 = self.c6_a(u6)
        c6 = self.c6_b(c6)

        u7 = self.tc7(c6)
        u7 = tf.concat([u7, c3], axis=3)
        u7 = self.drop7(u7)
        c7 = self.c7_a(u7)
        c7 = self.c7_b(c7)

        u8 = self.tc8(c7)
        u8 = tf.concat([u8, c2], axis=3)
        u8 = self.drop8(u8)
        c8 = self.c8_a(u8)
        c8 = self.c8_b(c8)

        u9 = self.tc9(c8)
        u9 = tf.concat([u9, c1], axis=3)
        u9 = self.drop9(u9)
        c9 = self.c9_a(u9)
        c9 = self.c9_b(c9)
        outputs = self.outconv(c9)

        return outputs


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
        self.GraspUNet = GraspUNet()
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

        ## Celui sur lequel les tests sont faits
        x = self.QGraspTest(self.Dense(input))
        ## Test zone
        # x = self.GraspUNet(input)
        # return pos, cos, sin, width
        return x

if __name__ == "__main__":
    im = np.ndarray((1, 224, 224, 3), np.float32)
    Densenet = Reinforcement()
    print(Densenet(im).shape)
