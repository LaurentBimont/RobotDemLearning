import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def angle2robotangle(angle):
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    angle -= 180
    return angle

def preprocess_depth_img(depth_image):
    min = second_min(depth_image.flatten())
    depth_image = np.ones(depth_image.shape) - (depth_image - min) / (depth_image.max() - min)
    depth_image[depth_image > 1] = 0
    depth_image = (depth_image * 255).astype('uint8')
    depth_image = np.asarray(np.dstack((depth_image, depth_image, depth_image)), dtype=np.uint8)
    return depth_image

def second_min(vector):
    A = sorted(vector)
    print(len(vector))
    for i in range(len(vector)):
        if A[i] != 0:
            print(A[i], i)
            return A[i]
    return 0

def rotate_image2(input_data, input_angles):
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

def postprocess_pred(out):
    out[out < 0] = 0
    zoom_pixel = 30

    (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
    test_pca = out[y_max-zoom_pixel:y_max+zoom_pixel, x_max-zoom_pixel:x_max+zoom_pixel, 1]
    PointCloud = heatmap2pointcloud(test_pca)
    pca = PCA()
    pca.fit(PointCloud)
    vectors = pca.components_
    sing_val = pca.singular_values_/np.linalg.norm(pca.singular_values_)
    vectors[0] *= sing_val[0]
    vectors[1] *= sing_val[1]
    np.linalg.norm(pca.singular_values_)
    origin = [zoom_pixel], [zoom_pixel]
    e = 30
    theta = py_ang([1, 0], vectors[0])*180/np.pi
    return x_max, y_max, theta, e
