import cv2
import numpy as np
import sys

def inhouse_bilateral(image_, spatial_variance, intensity_variance, kernel_size):
    image_2 = np.zeros(image_.shape)
    gaussKer = cv2.getGaussianKernel(ksize=kernel_size, sigma=spatial_variance)
    sizeX, sizeY = image_.shape
    sigma = np.sqrt(intensity_variance)
    half = kernel_size // 2
    cons = 1 / (sigma * np.sqrt(2 * np.pi))
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):
            image_S = image_[i - half : i + half + 1, j - half : j + half + 1]
            image_I = image_S - image_S[kernel_size // 2, kernel_size // 2]
            image_IG = cons * np.exp(-((image_I / sigma) ** 2) * 0.5)
            weights = np.multiply(gaussKer, image_IG)
            vals = np.multiply(image_S, weights)
            val = np.sum(vals) / np.sum(weights)
            image_2[i, j] = val
    return image_2
 

# image = cv2.imread("Image Processing/EE604 A3/noisy1.jpg")
input_file = sys.argv[1]
image = cv2.imread(input_file)
cv2.imshow("input image", image)

out = (image/255.0).astype("float32")
rgb_planes = cv2.split(out)
rgb_new_pln = []
for plane in rgb_planes:
    out_pln = inhouse_bilateral(plane, 60, 0.0035, 21)
    rgb_new_pln.append(np.uint8(out_pln*255))
res = cv2.merge(rgb_new_pln)
pad = image.copy()
pad[10:-10, 10:-10, :] = 0
res = res + pad

status = cv2.imwrite('denoised.jpg', res)
print("Image written to file-system : ", status)
cv2.imshow("output image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()