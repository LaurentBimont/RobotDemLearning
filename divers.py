import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.transform import resize
import cv2

def heatmap2pointcloud(img):
    # Rescale between 0 and 1
    plt.imshow(img)
    print('Ce qui rentre ')
    plt.show()
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    PointCloudList = []
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
        print(1)
        angle -= 180
    elif angle < -90:
        print(2)
        # angle += 180
    angle -= 180

    return angle

def preprocess_depth_img(depth_image):
    # depth_image[depth_image > 0.55] = 0 Commenté le 16 juillet
    depth_image[depth_image > 0.45] = 0  # Rajouté le 16 juillet
    depth_image[depth_image < 0.35] = 0
    # depth_image[depth_image == 0] = np.mean(depth_image[depth_image != 0])  Commenté le 16 juillet
    depth_image[depth_image == 0] = np.max(depth_image[depth_image != 0])
    print('C est par la', np.max(depth_image), np.min(depth_image), np.mean(depth_image))
    min = second_min(depth_image.flatten())
    mini = np.min(depth_image.flatten())
    print('Le deuxième minimum est ', min)
    plt.subplot(1, 3, 1)
    plt.imshow(depth_image)
    depth_image = np.ones(depth_image.shape) - (depth_image - mini) / (depth_image.max() - mini)
    plt.subplot(1, 3, 2)
    plt.imshow(depth_image)
    # depth_image[depth_image > 1] = 0
    plt.subplot(1, 3, 3)
    plt.imshow(depth_image)
    plt.show()
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

def postprocess_pred(out, camera):
    out[out < 0] = 0
    zoom_pixel = 100

    (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
    ####### REMARQUE quand le max pixel est a moins de 50 des bords, ca buggue ######
    test_pca = out[max(y_max-zoom_pixel, 0):min(y_max+zoom_pixel, out.shape[0]), max(x_max-zoom_pixel, 0):min(x_max+zoom_pixel, out.shape[1]), 1]
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
    # e_mm = get_ecartement_pince(sing_val[1], theta, (y_max, x_max), camera)
    return x_max, y_max, theta, e


def get_ecartement_pince(vp, theta, center, camera):
    sigma = 2*np.sqrt(vp)
    v0, u0 = center

    P0 = camera.transform_3D(u0, v0)

    u1 = int(u0 - sigma * np.cos(theta))
    v1 = int(v0 + sigma * np.sin(theta))
    P1 = camera.transform_3D(u1,v1)

    e = np.sqrt((P0-P1).dot(P0-P1))

    alpha = 1  

    return e*alpha
    #  heightmap = im.copy()
   #  im2 = heightmap[:,:,1]
   #  im2 = (im2-np.min(im2))/(np.max(im2)-np.min(im2))
   #  im2 = im2*255
   #  im2 = im2.astype(np.uint8)

   #  ret,thresh = cv2.threshold(im2,60,255,0)
   #  contours,hierarchy = cv2.findContours(thresh, 1, 2)
   #  cnt = contours[0]
   #  rect = cv2.minAreaRect(cnt)
   #  box = cv2.boxPoints(rect)
   #  box = np.int0(box)
   #  cv2.drawContours(heightmap,[box],0,(0,0,255),2)
   #  ellipse = cv2.fitEllipse(cnt)

   #  cv2.ellipse(heightmap,ellipse,(0,255,0),2)
   #  cv2.drawContours(heightmap,[box],0,(0,0,255),2)

   #  plt.imshow(heightmap)
   #  plt.show()
   #  a = (ellipse[0][0]-ellipse[1][0])/2
   #  b = (ellipse[0][1]-ellipse[1][1])/2
   #  a = min(a,b)

   # return a 

def get_angle(fingers):
    u1, v1, u2, v2 = fingers[0], fingers[1], fingers[2], fingers[3]
    angle = 180/np.pi * py_ang(np.array([u1-u2, v1-v2]), np.array([1, 0]))
    return(angle - 90)

def get_ecartement(fingers):
    # return np.linalg.norm(self.cam.transform_3D(u1, v1)-self.cam.transform_3D(u2, v2))
    u1, v1, u2, v2 = fingers[0], fingers[1], fingers[2], fingers[3]
    ##### Temporaire : renvoie juste la difference en pixel #######"
    return np.linalg.norm(np.array([[u1-u2], [v1-v2]]))

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


def compute_labels(best_pix_ind, shape=(224,224,3), viz=False):
    '''Create the targeted Q-map
    :param label_value: Reward of the action
    :param best_pix_ind: (Rectangle Parameters : x(colonne), y(ligne), angle(en degré), ecartement(en pixel)) Pixel where to perform the action
    :return: label : an 224x224 array where best pix is at future reward value
             label_weights : a 224x224 where best pix is at one
    '''
    label = np.zeros(shape, dtype=np.float32)
    for i in range(len(best_pix_ind)):
        label_temp = np.zeros(shape, dtype=np.float32)
        x, y, angle, e, lp, label_val = best_pix_ind[i]
        print(best_pix_ind[i])
        rect = draw_rectangle(e, angle, x, y, lp)
        cv2.fillConvexPoly(label_temp, rect, color=(255, 255, 255))
        label_temp[np.where(label_temp == 255)] = label_val
        print('le minimax de label temp', np.min(label_temp), np.max(label_temp))
        label = label + label_temp
        print('Label', np.min(label), np.max(label))
    print(np.min(label), np.max(label))
    label = resize(label, shape)
    print('Le label final')
    plt.imshow(label)
    plt.show()
    # label_test = (label - np.min(label))/(np.max(label)-np.min(label))
    # label_test = resize(label_test, (224, 224, 3))
    # plt.imshow(label_test)
    # plt.show()
    return label
