########## Trainer.py class Trainer ###########

def max_primitive_pixel(self, prediction, viz=False):
    '''Locate the max value-pixel of the image
    Locate the highest pixel of a Q-map
    :param prediction: Q map
    :return: max_primitive_pixel_idx (tuple) : pixel of the highest Q value
             max_primitive_pixel_value : value of the highest Q-value
    '''
    # Transform the Q map tensor into a 2-size numpy array
    numpy_predictions = prediction.numpy()[0, :, :, 0]
    if viz:
        result = tf.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
        plt.subplot(1, 2, 1)
        plt.imshow(result)
        plt.subplot(1, 2, 2)
        plt.imshow(numpy_predictions)
        plt.show()
    # Get the highest pixel
    max_primitive_pixel_idx = np.unravel_index(np.argmax(numpy_predictions),
                                               numpy_predictions.shape)
    # Get the highest score
    max_primitive_pixel_value = numpy_predictions[max_primitive_pixel_idx]
    print('Grasping confidence scores: {}, {}'.format(max_primitive_pixel_value, max_primitive_pixel_idx))
    return max_primitive_pixel_idx, max_primitive_pixel_value

def get_best_predicted_primitive(self):
    '''
    :param output_prob: Q-map
    :return: best_idx (tuple): best idx in raw-Q-map
             best_value : highest value in raw Q-map
             image_idx (tuple): best pixels in image format (224x224) Q-map
             image_value : best value in image format (224x224) Q-map
    '''

    # Best Idx in image frameadients(
    prediction = tf.image.resize_images(self.output_prob, (224, 224))
    image_idx, image_value = self.max_primitive_pixel(prediction)
    # Best Idx in network output frame
    best_idx, best_value = self.max_primitive_pixel(self.output_prob)

    self.best_idx, self.future_reward = best_idx, best_value
    return best_idx, best_value, image_idx, image_value

#### Main.py
def grasping_rectangle_error(params):
    global out, x_max, y_max, lp
    img = out
    rect = draw_rectangle(img, params, x_max, y_max, lp)
    mask = cv2.fillConvexPoly(np.zeros(img.shape[:2]), rect, color=1)
    masked_img = img[:, :, 1] * mask
    score = (np.sum(masked_img)**3)/(np.sum(mask)**2)
    return -score

##### FingerTracking.py

def detect_green_viz(self):
    '''
    For Vizualisation only of the
    :return:
    '''
    first_cont, second_cont = None, None

    # Min Distance over a period of 5 seconds
    if time.time() - self.t0 > 10:
        self.t0 = time.time()
        self.min_over_time = np.inf

    # Take each frame
    self.cam.get_frame()
    frame = self.cam.color_image
    _, frame = self.cam.get_frame()
    # Convert BGR to HSV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # define range for green color in HSV
    lower_green = np.array([60, 40, 40])
    upper_green = np.array([90, 250, 250])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    first_cont, second_cont = self.max_contour(cont)
    if first_cont is not None and second_cont is not None:
        cx1, cy1 = self.centroid(first_cont)
        cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)

        cx2, cy2 = self.centroid(second_cont)
        cv2.circle(frame, (cx2, cy2), 5, [0, 255, 0], -1)

        xtcp, ytcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2

        print(FT.get_ecartement(cx1, cy1, cx2, cy2), self.min_over_time)

        if FT.get_ecartement(cx1, cy1, cx2, cy2) < self.min_over_time:
            self.x_tcp, self.y_tcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2
            self.min_over_time = FT.get_ecartement(cx1, cy1, cx2, cy2)
        cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)
    else:
        cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)
    # except Exception as e:
    #     print(str(e))
    #     print('Pas de contours')
    #     pass

    return frame, mask, res


##### Paramètres Intrinsèques
    # trouver les bons paramètres intrinsèques
#
#     P1_robot, P2_robot, P3_robot = np.array([417.7, 143.15, 136.22]),\
#                                    np.array([405.81, 229.73, 134.35]),\
#                                    np.array([559.89, 117.24, 135.45])       # Position des points dans le repère robot
#
#     print('distance entre 1 et 2', (np.linalg.norm(P1_robot-P2_robot) - np.linalg.norm(P1_camera-P2_camera)))
#     print('distance entre 3 et 2 ', (np.linalg.norm(P2_robot - P3_robot) - np.linalg.norm(P2_camera - P3_camera)))
#     print('distance entre 1 et 3', (np.linalg.norm(P1_robot - P3_robot) - np.linalg.norm(P1_camera - P3_camera)))
#
#     P_robot = np.array([np.transpose(P1_robot), np.transpose(P2_robot), np.transpose(P3_robot)])
#
#     def error_intrinsic(param):
#         P1_robot, P2_robot, P3_robot = np.array([417.7, 143.15, 136.22]), \
#                                        np.array([405.81, 229.73, 134.35]), \
#                                        np.array([559.89, 117.24, 135.45])  # Position des points dans le repère robot
#         global P1, P2, P3
#         P1_camera = Cam.transform_3D(int(P1[0][0]), int(P1[0][1]), image=camera_depth_img, param=param)
#         P3 = corners2[47]
#         P3_camera = Cam.transform_3D(int(P3[0][0]), int(P3[0][1]), image=camera_depth_img, param=param)
#         P2 = corners2[0]
#         P2_camera = Cam.transform_3D(int(P2[0][0]), int(P2[0][1]), image=camera_depth_img, param=param)
#
#         error = (np.linalg.norm(P1_robot-P2_robot) - np.linalg.norm(P1_camera-P2_camera))**2 + \
#                 (np.linalg.norm(P2_robot - P3_robot) - np.linalg.norm(P2_camera - P3_camera))**2 + \
#                 (np.linalg.norm(P1_robot - P3_robot) - np.linalg.norm(P1_camera - P3_camera))**2
#         return error
#
# arg_opt = [Cam.intr.fx, Cam.intr.fy]
#
# optim_result = optimize.minimize(error_intrinsic, arg_opt, method='Nelder-Mead')
# result = optim_result.x
#
# print(result, arg_opt)
# P1_camera = Cam.transform_3D(int(P1[0][0]), int(P1[0][1]), image=camera_depth_img, param=result)
# P3 = corners2[47]
# P3_camera = Cam.transform_3D(int(P3[0][0]), int(P3[0][1]), image=camera_depth_img, param=result)
# P2 = corners2[0]
# P2_camera = Cam.transform_3D(int(P2[0][0]), int(P2[0][1]), image=camera_depth_img, param=result)
# print('distance entre 1 et 2', (np.linalg.norm(P1_robot-P2_robot) - np.linalg.norm(P1_camera-P2_camera)))
# print('distance entre 3 et 2 ', (np.linalg.norm(P2_robot - P3_robot) - np.linalg.norm(P2_camera - P3_camera)))
# print('distance entre 1 et 3', (np.linalg.norm(P1_robot - P3_robot) - np.linalg.norm(P1_camera - P3_camera)))


######## calibration.py

def erreur(P1_camera2, P2_camera2, P3_camera2, P1_robot, P2_robot, P3_robot):
    erreur_globale = 0
    erreur_globale += (np.linalg.norm(P1_camera2-P2_camera2) - np.linalg.norm(P1_robot-P2_robot))**2
    erreur_globale += (np.linalg.norm(P1_camera2-P3_camera2) - np.linalg.norm(P1_robot-P3_robot))**2
    erreur_globale += (np.linalg.norm(P3_camera2-P2_camera2) - np.linalg.norm(P3_robot-P2_robot))**2
    return erreur_globale

def error_scaling(R):
    a, b = R[0], R[1]
    global P1_robot, P2_robot, P3_robot, P1_camera2, P2_camera2, P3_camera2
    P1_camera2_copy, P2_camera2_copy, P3_camera2_copy = np.copy(P1_camera2), np.copy(P2_camera2), np.copy(P3_camera2)
    P1_camera2_copy[0] = P1_camera2_copy[0]/a
    P2_camera2_copy[0] = P2_camera2_copy[0]/a
    P3_camera2_copy[0] = P3_camera2_copy[0]/a

    P1_camera2_copy[1] = P1_camera2_copy[1] / b
    P2_camera2_copy[1] = P2_camera2_copy[1] / b
    P3_camera2_copy[1] = P3_camera2_copy[1] / b

    error = erreur(P1_camera2_copy, P2_camera2_copy, P3_camera2_copy, P1_robot, P2_robot, P3_robot)
    rmse = np.sqrt(error)
    return rmse

arg_opt = np.array([1, 1])

optim_result = optimize.minimize(error_scaling, arg_opt, method='Nelder-Mead')
R = optim_result.x
a, b = R[0], R[1]
print(a, b)

print('Distance pt1 et pt2 (camera 2)', np.linalg.norm(P1_camera2-P2_camera2))
print('Distance pt1 et pt2 (camera 1)', np.linalg.norm(P1_camera1-P2_camera1))
print('Distance pt1 et pt2 (robot)', np.linalg.norm(P1_robot-P2_robot))

print('Distance pt1 et pt3 (camera 2)', np.linalg.norm(P1_camera2-P3_camera2))
print('Distance pt1 et pt3 (camera 1)', np.linalg.norm(P1_camera1-P3_camera1))
print('Distance pt1 et pt3 (robot)', np.linalg.norm(P1_robot-P3_robot))

print('Distance pt3 et pt2 (camera 2)', np.linalg.norm(P3_camera2-P2_camera2))
print('Distance pt3 et pt2 (camera 1)', np.linalg.norm(P3_camera1-P2_camera1))
print('Distance pt3 et pt2 (robot)', np.linalg.norm(P3_robot-P2_robot))
#
# P1_camera2[0] = P1_camera2[0]/a
# P2_camera2[0] = P2_camera2[0]/a
# P3_camera2[0] = P3_camera2[0]/a
#
# P1_camera2[1] = P1_camera2[1]/b
# P2_camera2[1] = P2_camera2[1]/b
# P3_camera2[1] = P3_camera2[1]/b
#
# P1_camera1[0] = P1_camera1[0]/a
# P2_camera1[0] = P2_camera1[0]/a
# P3_camera1[0] = P3_camera1[0]/a
#
# P1_camera1[1] = P1_camera1[1]/b
# P2_camera1[1] = P2_camera1[1]/b
# P3_camera1[1] = P3_camera1[1]/b

print('Distance pt1 et pt2 (camera 2)', np.linalg.norm(P1_camera2-P2_camera2))
print('Distance pt1 et pt2 (camera 1)', np.linalg.norm(P1_camera1-P2_camera1))
print('Distance pt1 et pt2 (robot)', np.linalg.norm(P1_robot-P2_robot))

print('Distance pt1 et pt3 (camera 2)', np.linalg.norm(P1_camera2-P3_camera2))
print('Distance pt1 et pt3 (camera 1)', np.linalg.norm(P1_camera1-P3_camera1))
print('Distance pt1 et pt3 (robot)', np.linalg.norm(P1_robot-P3_robot))

print('Distance pt3 et pt2 (camera 2)', np.linalg.norm(P3_camera2-P2_camera2))
print('Distance pt3 et pt2 (camera 1)', np.linalg.norm(P3_camera1-P2_camera1))
print('Distance pt3 et pt2 (robot)', np.linalg.norm(P3_robot-P2_robot))



########## Passage d'une repère à l'autre ###########
import numpy as np

def fromAtoB(PA, Ra2b, Ta2b_basea):
    '''
    Fait les calculs suivants : Pb = Ra2b.T x Pa - Ra2b.T x Ta2b_basea
    :param PA: Coordonné d'un point dans le repère A
    :param Ra2b: Rotation de A vers B
    :param Ta2b_basea: Vecteur de translation OaOb exprimé dans la base A
    :return: PB : Coordonnées du point P dans le repère B
    '''

    T = np.transpose(Ra2b).dot(np.transpose(Ta2b_basea))
    print(np.transpose(T))
    PB = np.transpose(Ra2b).dot(np.transpose(PA)) - np.transpose(T)
    return PB

def rotation_matrix(theta):
    theta = np.pi/180 * theta
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

R = rotation_matrix(90)
PA = np.array([1, 0, 0])

# T est exprimé dans la base de départ
print(fromAtoB(PA, R, np.array([6, 1, 0])))

## trainer.py

def main_augmentation(self, dataset):
    ima, val, val_w = dataset['im'], dataset['label'], dataset['label_weights']
    self.future_reward = 1
    for j in range(len(ima)):
        if j % 10 == 0:
            print('Iteration {}/{}'.format(j, len(ima)))
        with tf.GradientTape() as tape:
            self.forward(tf.reshape(ima[j], (1, 224, 224, 3)))
            self.compute_loss_dem(val[j], val_w[j])
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())

def main(self, input):
    self.future_reward = 1
    with tf.GradientTape() as tape:
        self.forward(input)
        self.compute_loss()
        grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                       global_step=tf.train.get_or_create_global_step())

def output_viz(self, output_prob):
    output_viz = np.clip(output_prob, 0, 1)
    output_viz = cv2.applyColorMap((output_viz*255).astype(np.uint8), cv2.COLORMAP_JET)
    output_viz = cv2.cvtColor(output_viz, cv2.COLOR_BGR2RGB)
    return np.array([output_viz])


def compute_loss(self):
    # A changer pour pouvoir un mode démonstration et un mode renforcement
    expected_reward, action_reward = self.action.compute_reward(self.action.grasp, self.future_reward)
    label224, label_weights224 = self.compute_labels(expected_reward, self.best_idx)
    label, label_weights = self.reduced_label(label224, label_weights224)
    self.output_prob = tf.reshape(self.output_prob, (self.width, self.height, 1))
    self.loss_value = self.loss(label, self.output_prob, label_weights)
    return self.loss_value


def backpropagation(self, gradient):
    self.optimizer.apply_gradients(zip(gradient, self.myModel.trainable_variables),
                                   global_step=tf.train.get_or_create_global_step())
    self.iteration = tf.train.get_global_step()


## Finger Tracking
def createHistogram(self):
    t0 = time.time()
    width = 50
    while (time.time() - t0)<5:
        _, frame = self.cam.get_frame()
        rows, cols, _ = frame.shape
        y0, x0 = int(0.5 * rows), int(0.2 * cols)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (x0, y0), (x0+width, y0+width), (255,0,0), 2)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    rows, cols, _ = frame.shape
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([width, width], dtype=hsvFrame.dtype)
    y0, x0 = int(0.5 * rows), int(0.2 * cols)
    roi = hsvFrame[y0:y0 + width, x0:x0 + width]
    hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

def histMasking(self, frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.dilate(thresh, kernel, iterations=5)
    # Erode : Only if all pixel under mask are 1
    # Dilate : If at least one pixel under mask is 1
    # Opening : Erosion then Dilatation
    # Closing : Dilatation then Erosion

    thresh = cv2.erode(thresh, kernel, iterations=5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)

# A mettre dans div
def get_angle(self, fingers):
    u1, v1, u2, v2 = fingers[0], fingers[1], fingers[2], fingers[3]
    angle = 180/np.pi * div.py_ang(np.array([u1-u2, v1-v2]), np.array([1, 0]))
    return(angle - 90)
