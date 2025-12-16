import cv2
import numpy as np
import sys

input_file = sys.argv[1]
image = cv2.imread(input_file)
# image = cv2.imread("Image Processing/EE604 A2/cctv4.jpg")
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', img)

def ginject(gamma, image_):
    g_img = np.zeros((image_.shape), np.uint8)
    for x in range(image_.shape[1]):
        for y in range(image_.shape[0]):
            g_img[y][x]=255*(image_[y][x]/255)**gamma
    return g_img

g_img = ginject(0.4, img)
equalized_img = cv2.equalizeHist(img)
new_img = cv2.addWeighted(equalized_img, 0.45, g_img, 0.65, 0)

cv2.imshow('Enhanced', new_img)
name = 'enhanced-cctv' + str(input_file)[-5] + '.jpg'
status = cv2.imwrite(name, new_img)
print("Image written to file-system : ", status)
cv2.waitKey(0)
cv2.destroyAllWindows()