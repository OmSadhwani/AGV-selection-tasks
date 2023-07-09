
from sklearn.linear_model import LinearRegression, RANSACRegressor

import numpy as np
import matplotlib.pyplot as plt

import cv2

if __name__ == '__main__':

    img = cv2.imread('line_ransac.png')
    cv2.imshow("original image",img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray, (3,3), 0)
    ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = np.zeros(img.shape)
    # for i in range(len(contours)):
        # cv2.drawContours(image,contours, i ,(255,0,0),-1)
    X1=[]
    y1=[]

    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        X1.append(cX)
        y1.append(cY)


    X=np.array(X1).reshape(-1,1)
    y=np.array(y1).reshape(-1,1)






    # plt.imshow(image)




    ransac = RANSACRegressor(base_estimator=LinearRegression(),
                             min_samples=2, max_trials=100,
                             loss='absolute_loss', random_state=42,
                             residual_threshold=10)

    ransac.fit(X, y)



    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)




    plt.figure(figsize=(8, 8))
    plt.scatter(X[inlier_mask], y[inlier_mask],
                c='blue', edgecolor='white',
                marker='o', label='Inliers')

    plt.scatter(X[outlier_mask], y[outlier_mask],
                c='green', edgecolor='white',
                marker='s', label='Outliers')

    line_X = np.arange(3, 700, 5)


    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    plt.plot(line_X, line_y_ransac, color='red', lw=2)

    plt.legend(loc='upper left', fontsize=12)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()