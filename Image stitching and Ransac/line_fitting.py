import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

import cv2

def dist(x0,y0,x1,y1,x2,y2,t):
        if x1!=x2 :
            m= (y1-y2)/(x1-x2)
            if (abs(y0-y1-m*(x0-x1))/sqrt((1+m**2))<=t):
                return True
        elif (abs(x0-x1)<=t) :
            return True
        else : return False


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
    Y1=[]

    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        X1.append(cX)
        Y1.append(cY)

# plt.imshow(image)

mx=0
ct=0
corx=[]
cory=[]
t=5
ly=0
lx=0
mxx=0
my=0
inliersx=[]
inliersy=[]

for x,y in zip(X1,Y1):
    for x1,y1 in zip(X1,Y1):
        if x==x1 and y==y1 :
            continue

        # m= (y-y1)/(x-x1)
        for x0,y0 in zip(X1,Y1):
            if (dist(x0,y0,x1,y1,x,y,t)):
                    ct=ct+1


        if (ct>mx):
            mx=ct
            if(x!=x1):
                m=(y-y1)/(x-x1)
            corx.clear()
            cory.clear()
            corx.append(x)
            corx.append(x1)
            cory.append(y)
            cory.append(y1)
            ly=y
            lx=x
            mxx=x1
            my=y1
        ct=0

for x0,y0 in zip(X1,Y1):
    if(dist(x0,y0,lx,ly,mxx,my,t)):
        inliersx.append(x0)
        inliersy.append(y0)



corx.append(0)
corx.append(700)
cory.append(ly+m*(0-lx))
cory.append(ly+m*(700-lx))

plt.figure(figsize=(8, 8))
plt.scatter(X1, Y1,
                c='red', edgecolor='white',
                marker='s',label='outliers')
plt.scatter(inliersx, inliersy,
                c='blue', edgecolor='white',
                marker='o',label="inliers")

plt.plot(corx,cory, color='red', lw=2)

plt.legend(loc='upper left', fontsize=12)

ax = plt.gca()
ax.invert_yaxis()
plt.show()