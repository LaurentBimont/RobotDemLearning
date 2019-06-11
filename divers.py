import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def draw_rectangle(e, theta, x0, y0, lp):
    #x1, y1, lx, ly, theta = params[0], params[1], params[2], params[3], params[4]
    theta_rad = theta * np.pi/180
    x1 = int(x0 - lp/2*np.cos(theta_rad) - e/2*np.sin(theta_rad))
    y1 = int(y0 + lp/2*np.sin(theta_rad) - e/2*np.cos(theta_rad))
    x2 = int(x0 + lp/2*np.cos(theta_rad) - e/2*np.sin(theta_rad))
    y2 = int(y0 - lp/2*np.sin(theta_rad) - e/2*np.cos(theta_rad))
    x3 = int(x0 - lp/2*np.cos(theta_rad) + e/2*np.sin(theta_rad))
    y3 = int(y0 + lp/2*np.sin(theta_rad) + e/2*np.cos(theta_rad))
    x4 = int(x0 + lp/2*np.cos(theta_rad) + e/2*np.sin(theta_rad))
    y4 = int(y0 - lp/2*np.sin(theta_rad) + e/2*np.cos(theta_rad))

    return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int)

def heatmap2pointcloud(img):
    # Rescale between 0 and 1
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    PointCloudList = []
    img = img - 0.6
    img[img<0] = 0.
    plt.imshow(img)
    plt.show()
    for index, x in np.ndenumerate(img):
        for i in range(int(x*10)):
            PointCloudList.append([index[1], 100-index[0]])

    return np.asarray(PointCloudList)

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))

    det = v1[0] * v2[1] - v1[1] * v2[0]
    if det<0:
        return -np.arctan2(sinang, cosang)
    else:
        return np.arctan2(sinang, cosang)



def rotate_image2( input_data, input_angles):
    return tf.contrib.image.rotate(input_data, input_angles, interpolation="BILINEAR")


def preprocess_img(img, target_height=224*5, target_width=224*5, rotate=False):
    # Apply 2x scale to input heightmaps
    resized_img = tf.image.resize_images(img, (target_height, target_width))

    # Peut être rajouter un padding pour éviter les effets de bords

    if rotate:
        rimgs = rotate_image2(resized_imgs, list_angles)

        # Add extra padding (to handle rotations inside network)
        diag_length = float(target_height) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - target_height))

        padded_imgs = tf.image.resize_image_with_crop_or_pad(rimgs,target_height+padding_width,target_width+padding_width)

        return padded_imgs, padding_width

    return resized_img


def postprocess_img( imgs, list_angles):
    # Return Q values (and remove extra padding
    # Reshape to standard
    resized_imgs = tf.image.resize_images(imgs, (320, 320))
    # Perform rotation
    rimgs = rotate_image2(resized_imgs, list_angles)
    # Reshape rotated images
    resized_imgs = tf.image.resize_images(rimgs, (320, 320))
    return resized_imgs
