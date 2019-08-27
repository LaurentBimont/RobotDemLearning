import numpy as np
import matplotlib.pyplot as plt
import cv2
import divers as div
import scipy.ndimage as scipy_image


#Test
#from trainer import Trainer

class OnlineAugmentation(object):
    def __init__(self):
        self.batch = None
        self.general_batch = {'im': [], 'label': []}
        self.im, self.label = 0, 0
        self.seed = np.random.randint(1234)
        self.original_size = (224, 224)

    def add_im(self, img, label):
        '''
        Add a batch composed of (image, label, label_weights) into the general batch dict
        :param batch: batch
        :return: None
        '''
        self.general_batch['im'].append(img.astype(np.float32))
        self.general_batch['label'].append(label.astype(np.float32))

    def generate_batch(self, im, label, mini, augmentation_factor=2, viz=False):
        '''Generate new images and label from one image/demonstration
        :param im: input image (numpy (224x224x3))
        :param label: label (numpy (224x224x3))
        :param augmentation_factor: number of data at the end of augmentation (3xaugmentation_factor³)
        :return: Batch of the augmented DataSet
        '''
        im, label = im.astype(np.uint8), label.astype(np.uint8)
        x, y, w, h = detect_tool(im[:, :, 0])
        square_im, square_lab = create_sub_square(im, label, x, y, w, h)

        x_lim, y_lim, _ = im.shape
        for i in range(0, augmentation_factor**3):
            xc, yc, angle = int(np.random.random()*x_lim), int(np.random.random()*y_lim), int(360*np.random.random())
            bool_add_noise = np.random.random()<0.6
            if bool_add_noise:
                noisy_square_im = add_noise(np.copy(square_im))
            else:
                noisy_square_im = np.copy(square_im)

            rot_plain_img, rot_plain_label = rotation_insert(noisy_square_im, square_lab, angle, xc, yc, x_lim, y_lim)
            if bool_add_noise:
                rot_plain_img = add_noise(rot_plain_img)
            rot_plain_label = change_ang_value(rot_plain_label, angle)
            self.add_im(rot_plain_img, rot_plain_label)

            up_down_flip_label = flip_ang_value(np.copy(rot_plain_label), 1)
            self.add_im(np.flip(rot_plain_img, 0), np.flip(up_down_flip_label, 0))

            left_right_flip_label = flip_ang_value(np.copy(rot_plain_label), 2)
            self.add_im(np.flip(rot_plain_img, 1), np.flip(left_right_flip_label, 1))
            # plt.imshow(np.flip(left_right_flip_label, 1))
            # plt.show()
            both_flip_label = flip_ang_value(np.copy(rot_plain_label), 3)
            self.add_im(np.flip(np.flip(rot_plain_img, 1), 0), np.flip(np.flip(both_flip_label, 1), 0))

            # PETIT TRAVAIL A FAIRE SI ON VEUT RAJOUTER L'ANGLE
            if viz:
                plt.subplot(2, 2, 1)
                plt.imshow(rot_plain_img)
                plt.subplot(2, 2, 2)
                plt.imshow(rot_plain_label)

                plt.subplot(2, 2, 3)
                plt.imshow(np.flip(np.flip(rot_plain_img, 1), 0))
                plt.subplot(2, 2, 4)
                plt.imshow(np.flip(np.flip(rot_plain_label, 1), 0))
                plt.show()
        return self.general_batch

def add_noise(im):
    for i in range(5):
        # x, y = int(np.random.random()*im.shape[0]), int(np.random.random()*im.shape[1])
        # im[max(0, x-int(np.random.random()*10)):min(im.shape[0], x+int(np.random.random()*10)),
        # max(0, y-int(np.random.random()*10)):min(im.shape[0], y+int(np.random.random()*10))] = 255
        x, y = int(np.random.random() * im.shape[0]), int(np.random.random() * im.shape[1])
        im[max(0, x - int(np.random.random()*10)):min(im.shape[0], x + int(np.random.random()*10)),
        max(0, y - int(np.random.random()*10)):min(im.shape[0], y + int(np.random.random()*10))] = 0
    # plt.imshow(im)
    # plt.show()
    return im

def detect_tool(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = im.shape
    (x, y, w, h) = cv2.boundingRect(contours[np.argmax([len(i) for i in contours])])  # Sélectionne le contour le plus long
    x, y, w, h = div.evenisation(x - 10), div.evenisation(y - 10), div.evenisation(w + 10), div.evenisation(h + 10)
    viz = False
    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(im)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.subplot(1, 2, 2)
        plt.imshow(im)
        plt.show()
    return x, y, w, h

def create_sub_square(im, label, x, y, w, h):
    # Zoom des parties intéressantes à augmenter
    zoom_im, zoom_lab = im[y:y+h, x:x+w, :], label[y:y+h, x:x+w, :]
    # Mise dans un carré permettant des rotations sans aucun troncage sqrt(2)*c
    great_length = div.evenisation(int(np.sqrt(2)*np.max(zoom_im.shape)) + 1)

    square_im, square_lab = np.zeros((great_length, great_length, 3)),\
                            np.zeros((great_length, great_length, 3), dtype=np.int)


    square_im[great_length//2-h//2:great_length//2+h//2, great_length//2-w//2:great_length//2+w//2, :] = zoom_im[:, :, :]

    square_lab[great_length//2-h//2:great_length//2+h//2, great_length//2-w//2:great_length//2+w//2, :] = zoom_lab[:, :, :]
    print('ici')
    viz = False
    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(square_im)
        plt.subplot(1, 2, 2)
        plt.imshow(square_lab)
        plt.show()
    return square_im, square_lab

def insert_demo(img, xc, yc, x_lim, y_lim):
    '''
    :param img: XX.XX.3 numpy array
    :return: img inserted into another bigger frame
    '''
    plain_img = np.zeros((x_lim, y_lim, 3))
    great_length, c = img.shape[0], img.shape[0] // 2
    plain_img[max(0, xc - c):min(x_lim, xc + c), max(0, yc - c):min(y_lim, yc + c), :] = img[max(0, c - xc):min(
        x_lim + c - xc, great_length), max(0, c - yc):min(y_lim + c - yc, great_length), :]
    return plain_img

def rotation_insert(img, label, angle, xc, yc, x_lim, y_lim):
    label_pos, label_neg = np.copy(label[:, :, 0]), np.copy(label[:, :, 0])

    label_pos[label_pos == 255] = 0
    label_pos[label_pos != 0] = 255
    label_neg[label_neg != 255] = 0

    # label_pos, label_neg = label_pos.astype(np.uint32), label_neg.astype(np.uint32)
    label_pos = scipy_image.rotate(label_pos, angle, reshape=False)
    label_neg = scipy_image.rotate(label_neg, angle, reshape=False)

    rot_img, rot_label = scipy_image.rotate(img.astype(np.uint8), angle, reshape=False), scipy_image.rotate(label, angle,
                                                                                                            reshape=False)
    rot_label[:, :, 0] = np.zeros(rot_label[:, :, 0].shape)
    label_neg, label_pos = label_neg.astype(np.int32), label_pos.astype(np.int32)
    label_pos[label_pos > 0] = 1
    label_pos[label_pos < 0] = 1
    label_neg[label_neg != 0] = 255

    rot_label = rot_label.astype(np.int32)
    rot_label[:, :, 0] = label_pos + label_neg
    rot_img, rot_label = insert_demo(rot_img, xc, yc, x_lim, y_lim), insert_demo(rot_label, xc, yc, x_lim, y_lim)
    # rot_img[rot_img < 0.5] = 0
    # rot_label[rot_label < 0.5] = 0

    return rot_img.astype(np.int32), rot_label.astype(np.int32)

def change_ang_value(label, ang):

    square_lab_ang = label[:, :, 1]

    square_lab_ang[np.where(square_lab_ang != 0)] += ang
    label[:, :, 1] = square_lab_ang%180
    return label

def flip_ang_value(label, thetype):
    '''
    :param label:
    :param thetype: 1 : up and down
                    2 : left to right
                    3 : both
    '''
    if thetype == 1:
        # plt.subplot(1, 2, 1)
        # plt.imshow(label[:, :, 1])
        label[:, :, 1] = -label[:, :, 1]

    if thetype == 2:
        label[:, :, 1] = 180 - label[:, :, 1]
    if thetype == 3:
        label[:, :, 1] = 180 + label[:, :, 1]
    label[:, :, 1] = label[:, :, 1] % 180
    # plt.subplot(1, 2, 2)
    # plt.imshow(label[:, :, 1])
    # plt.show()
    return label

if __name__=="__main__":
    data = np.load('Experiences/Demonstration/depth_label/depth_parameters_demo0.npy')
    # img = (data[0, :, :, 0] > 2).astype(np.uint8)
    img = data[0, :, :, :].astype(np.uint8)
    label = data[1, :, :, :].astype(np.uint8)
    add_noise(img)
    x, y, w, h = detect_tool(img[:, :, 0])
    # Isolation de la zone comprenant la partie intéressante
    square_im, square_lab = create_sub_square(img, label, x, y, w, h)
    ang = 61
    square_lab_ang = change_ang_value(square_lab, ang)
    x_lim, y_lim, _ = img.shape
    xc, yc = 200, 200

    rot_plain_img, rot_plain_label = rotation_insert(square_im, square_lab, ang, xc, yc, x_lim, y_lim)

    # plt.subplot(1, 2, 1)
    # plt.imshow(rot_plain_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(rot_plain_label)
    # plt.show()

    DA = OnlineAugmentation()
    DA.generate_batch(img, label, 0, augmentation_factor=4)

    viz = False
    if viz:
        # plt.subplot(1, 2, 1)

        # plt.imshow(crop.numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(label_crop.numpy().astype(np.int))
        # plt.show()

        # plt.subplot(1, 2, 1)
        # plt.imshow(translate.numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(label_translate.numpy().astype(np.int))
        # plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(rotate.numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(label_rotate.numpy().astype(np.int))
        plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(flip.numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(label_flip.numpy().astype(np.int))
        plt.show()
