import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from camera import *
from cv2 import Canny
import cv2
import sklearn
from sklearn.decomposition import PCA


im=np.load("./outfrofrancois.npy")
plt.imshow(im)
plt.show()
im = im.copy()
im2 = im[:,:,1]
im2 = (im2-np.min(im2))/(np.max(im2)-np.min(im2))
im2 = im2*255
im2 = im2.astype(np.uint8)

ret,thresh = cv2.threshold(im2,60,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im,[box],0,(0,0,255),2)
ellipse = cv2.fitEllipse(cnt)

cv2.ellipse(im,ellipse,(0,255,0),2)
cv2.drawContours(im,[box],0,(0,0,255),2)

plt.imshow(im)
plt.show()
a = (ellipse[0][0]-ellipse[1][0])/2
b = (ellipse[0][1]-ellipse[1][1])/2
a = min(a,b)
print(min(a,b))

# def heatmap2pointcloud(img):
#     # Rescale between 0 and 1
#     plt.imshow(img)
#     print(img)
#     plt.show()
#     img = (img - np.min(img))/(np.max(img)-np.min(img))
#     PointCloudList = []
#     img = img - 0.6
#     img[img<0] = 0.
#     plt.imshow(img)
#     plt.show()
#     for index, x in np.ndenumerate(img):
#         for i in range(int(x*10)):
#             PointCloudList.append([index[1], 100-index[0]])

#     return np.asarray(PointCloudList)



# zoom_pixel = 60
# (y_max, x_max) = np.unravel_index(im[:, :, 1].argmax(), im[:, :, 1].shape)
# test_pca = im[y_max-zoom_pixel:y_max+zoom_pixel, x_max-zoom_pixel:x_max+zoom_pixel, 1]
# PointCloud = heatmap2pointcloud(test_pca)
# pca = PCA()
# pca.fit(PointCloud)



