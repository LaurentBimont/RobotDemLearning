import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.enable_eager_execution(config)

#Test
#from trainer import Trainer


class OnlineAugmentation(object):
    def __init__(self):
        self.batch = None
        self.general_batch = {'im': [], 'label': [], 'label_weights': []}
        self.im, self.label, self.label_weights = 0, 0, 0
        self.seed = np.random.randint(1234)
        self.original_size = (224, 224)

    def create_batch(self, im, label, label_weights):
        '''Create a tensorflow batch of image, label and label weights
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param label_weights: label weights (numpy (224x224x3))
        :return: Return a batch of those 3 inputs
        '''
        if type(im).__module__ == np.__name__:
            # If it is a numpy array
            im = np.reshape(im, (224, 224, 3))

        my_batch = [im, label, label_weights]
        self.batch = tf.stack(my_batch)

    def add_im(self, batch):
        '''
        Add a batch composed of (image, label, label_weights) into the general batch dict
        :param batch: batch
        :return: None
        '''
        self.general_batch['im'].append(batch[0])
        self.general_batch['label'].append(batch[1])
        self.general_batch['label_weights'].append(batch[2])

    def flip(self, im, label, label_weights):
        '''Flip images
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param label_weights: label weights (numpy (224x224x3))
        :return: Flipping inputs in Tensor Format (rq : flip[1] and flip[2] have to be resized to the output format
                 of the Network
        '''
        self.create_batch(im, label, label_weights)
        flip = tf.image.flip_up_down(self.batch)
        self.add_im(flip)
        flip = tf.image.flip_left_right(flip)
        self.add_im(flip)
        flip = tf.image.flip_left_right(self.batch)
        self.add_im(flip)

        # To be deleted
        return flip[0], flip[1], flip[2]

    def rotate(self, im, label, label_weights, angle=0):
        '''Rotate image
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param label_weights: label weights (numpy (224x224x3))
        :param angle: angle of rotation
        :return: Flipping inputs in Tensor Format (rq : rotation[1] and rotation[2] have to be resized to the output format
                 of the Network
        '''
        self.create_batch(im, label, label_weights)
        rotation = tf.contrib.image.rotate(self.batch, angles=angle)
        if self.assert_label(rotation[1]):
            self.add_im(rotation)

        # To be deleted
        return rotation[0], rotation[1], rotation[2]

    def crop(self, im, label, label_weights, zooming=224):
        '''Crop image
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param label_weights: label weights (numpy (224x224x3))

        :return: Cropping inputs in Tensor Format (rq : crop[1] and crop[2] have to be resized to the output format
                 of the Network
        '''
        self.create_batch(im, label, label_weights)
        x = tf.random_crop(self.batch, size=[3, zooming, zooming, 3], seed=self.seed)
        crop = tf.image.resize_images(x, size=self.original_size)
        if self.assert_label(crop[1]):
            self.add_im(crop)

        # To be deleted
        return crop[0], crop[1], crop[2]

    def translate(self, im, label, label_weights, pad_top=0, pad_left=0, pad_bottom=0, pad_right=0):
        '''Crop image
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param label_weights: label weights (numpy (224x224x3))
        :param pad_top: translation to the bottom (pixels)
        :param pad_left: translation to the right (pixels)
        :param pad_bottom: translation to the bottom (pixels)
        :param

        :return: Translated inputs in Tensor Format (rq : translate[1] and translate[2] have to be resized to the output format
                 of the Network
        '''
        self.create_batch(im, label, label_weights)
        height, width = 224, 224
        x = tf.image.pad_to_bounding_box(self.batch, pad_top, pad_left, height + pad_bottom + pad_top,
                                         width + pad_right + pad_left)
        # pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width
        translate = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)
        if self.assert_label(translate[1]):
            self.add_im(translate)

        # To be deleted
        return translate[0], translate[1], translate[2]

    def assert_label(self, label):
        '''Assert if a label image still contain the demonstrating grasping point
        :param label label: label(numpy (224x224x3))

        :return: True if a valid grasping point is still in the image
                 False otherwise
        '''
        if np.sum(label.numpy()) > 5:
            return True
        return False

    def generate_batch(self, im, label, label_weights, augmentation_factor=2, viz=False):
        '''Generate new images and label from one image/demonstration
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param label_weights: label weights (numpy (224x224x3))
        :param augmentation_factor: number of data at the end of augmentation (3xaugmentation_factor³)
        :return: Batch of the augmented DataSet
        '''
        h = 1
        print('taille image diametre ', im.shape)
        print('Facteur da ', augmentation_factor)
        for i in range(augmentation_factor):
            print(1)
            ima, lab, lab_w = self.crop(im, label, label_weights, zooming=np.random.randint(100, 200))

            if self.assert_label(lab):
                for j in range(augmentation_factor):
                    ima, lab, lab_w = self.translate(ima, lab, lab_w,
                                                     pad_top=np.random.randint(0, 50),
                                                     pad_left=np.random.randint(0, 50),
                                                     pad_bottom=np.random.randint(0, 50),
                                                     pad_right=np.random.randint(0, 50))
                    if self.assert_label(lab):
                        for k in range(augmentation_factor):
                            ima, lab, lab_w = self.rotate(ima, lab, lab_w, angle=np.random.rand()*0.785)
                            if viz:
                                if self.assert_label(lab):

                                    if h < 10 and k == 2:
                                        plt.figure(1)
                                        plt.subplot(3, 3, h)
                                        plt.imshow(ima.numpy())
                                        plt.figure(2)
                                        plt.subplot(3, 3, h)
                                        plt.imshow(lab.numpy())
                                        h += 1
                            if self.assert_label(lab):
                                self.flip(ima, lab, lab_w)
        if viz:
            plt.show()


        return self.general_batch


if __name__=="__main__":


    tf.enable_eager_execution()

    im = np.zeros((1, 224, 224, 3), np.float32)
    im[0, 70:190, 100:105, :] = 1
    im[0, 70:80, 80:125, :] = 1
    best_idx = [125, 103]
    print(-1)
    hey = Trainer()
    print(0)
    hey.forward(im)

    print(1)
    label, label_weights = hey.compute_labels(1.9, best_idx)
    print(2)
    OA = OnlineAugmentation()
    print(3)
    OA.create_batch(im, label, label_weights)
    print(4)
    flip, label_flip, label_weights_flip = OA.crop(im, label, label_weights, zooming=200)
    print(5)
    OA.generate_batch(im, label, label_weights)
    # Visualisation
    viz = True
    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(flip.numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(label_flip.numpy())
        plt.show()
