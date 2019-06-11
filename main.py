import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from scipy import optimize
import tensorflow as tf
from sklearn.decomposition import PCA
import divers as div

recompute = True

if recompute:
    from trainer import Trainer
    model = Trainer(load=True, snapshot_file='reference_good')
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    im[:, 150:160, 150:160, :] = 1
    im = tf.contrib.image.rotate(im, angles=45)
    model.forward(im)
    out = model.prediction_viz(model.output_prob, im)
else:
    out = np.load('trained_qmap.npy')

out[out < 0] = 0
zoom_pixel = 30

(y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
test_pca = out[y_max-zoom_pixel:y_max+zoom_pixel, x_max-zoom_pixel:x_max+zoom_pixel, 1]
PointCloud = div.heatmap2pointcloud(test_pca)
pca = PCA()
pca.fit(PointCloud)
vectors = pca.components_
sing_val = pca.singular_values_/np.linalg.norm(pca.singular_values_)
vectors[0] *= sing_val[0]
vectors[1] *= sing_val[1]
np.linalg.norm(pca.singular_values_)
origin = [zoom_pixel], [zoom_pixel]

print(vectors)

plt.subplot(1, 2, 1)
plt.scatter(PointCloud[:, 0], PointCloud[:, 1])
plt.quiver(*origin, vectors[0, 0], vectors[0, 1], color='r', scale=1)
plt.quiver(*origin, vectors[1, 0], vectors[1, 1], color='b', scale=1)
plt.subplot(1, 2, 2)
plt.imshow(test_pca[:, :])
plt.show()

e = 30
print(vectors)
theta = div.py_ang([1, 0], vectors[0])*180/np.pi

print('Valeur singulières :', pca.singular_values_)
print('Parametre du rectangle : ecartement {}, angle {}, x: {}, y: {}, longueur pince {}'.format(e, theta, x_max, y_max,20))

rect = div.draw_rectangle(e, theta, x_max, y_max, 20)
optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)


plt.subplot(2, 2, 1)
plt.imshow(optim_rectangle)
plt.subplot(2, 2, 2)
plt.imshow(out[:, :, 1] * optim_rectangle)
plt.subplot(2, 2, 3)
plt.imshow(out)
plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='yellow')
plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='yellow')

plt.subplot(2, 2, 4)
out[:, :, 0] = out[:, :, 0] * optim_rectangle
out[:, :, 1] = out[:, :, 1] * optim_rectangle
plt.imshow(out)
plt.show()
#

if __name__=="__main__":
    # (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
    # lp = 40
    # print(x_max, y_max)
    # params = [40, 40]    # [Ecartement pince, angle]
    # optim_result = optimize.minimize(grasping_rectangle_error, params, method='Nelder-Mead')
    # result = optim_result.x
    # rect = draw_rectangle(out, result, x_max, y_max, lp)
    # optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)

    test_rectangle = div.draw_rectangle(20, 20, 50, 150, 60)
    print(test_rectangle)
    autre_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), test_rectangle, color=1)

    # Dessin d'un rectangle
    # plt.imshow(autre_rectangle)
    # plt.show()
    # #
    # plt.subplot(2, 2, 1)
    # plt.imshow(optim_rectangle)
    # plt.subplot(2, 2, 2)
    # plt.imshow(out[:, :, 1] * optim_rectangle)
    # plt.subplot(2, 2, 3)
    # plt.imshow(out)
    # plt.subplot(2, 2, 4)
    # out[:, :, 0] = out[:, :, 0] * optim_rectangle
    # out[:, :, 1] = out[:, :, 1] * optim_rectangle
    # plt.imshow(out)
    # plt.show()

# Revoir la manière de calculer l'écartement
