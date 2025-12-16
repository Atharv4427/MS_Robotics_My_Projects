import numpy as np
import cv2
import sys

input_file = sys.argv[1]
image = cv2.imread(input_file)
# image = cv2.imread("Image Processing/EE604 A2/gutters3.jpg")
img = image.copy()
cv2.imshow('Original', img)

def dilate(d_img):
    dilated= np.zeros((d_img.shape), dtype=np.uint8)
    kernal= np.array([[0,1,0], [1,1,1],[0,1,0]])
    for i in range(1, d_img.shape[0]-1):
        for j in range(1,d_img.shape[1]-1):
            v_t= d_img[i-1:i+1+1, j-1:j+1+1]
            product= v_t*kernal
            dilated[i,j]= np.max(product)
    return dilated

rgb_planes = cv2.split(image)
result_planes = []
for plane in rgb_planes:
    dilated_img = dilate(plane)
    bg_img = cv2.medianBlur(dilated_img, 51)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)

res = cv2.merge(result_planes)
res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# sharp_kern = np.array([[0.2, -0.8, 0.2],
#                    [-0.8, 5.2,-0.8],
#                    [0.2, -0.8, 0.2]])/3

alpha = 1.4
beta = -185
res = cv2.addWeighted(res, alpha, img, 0.45, beta)

cv2.imshow('Final Image', res)
status = cv2.imwrite(' cleaned-gutter.jpg', res)
print("Image written to file-system : ", status)
cv2.waitKey(0)
cv2.destroyAllWindows()