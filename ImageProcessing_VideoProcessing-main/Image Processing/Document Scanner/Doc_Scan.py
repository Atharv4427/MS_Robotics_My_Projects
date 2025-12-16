import numpy as np
import cv2

# Read and display image
image = cv2.imread("Image Processing/20210610_180057.jpg")
image_copy = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
image_ = image_copy
def display(name, img, sizex, sizey):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, sizex, sizey)
    cv2.imshow(name, img)

display("Image", image, 604, 806)
#cv2.waitKey(0)

# Gausian Noise Reduction
image_copy = cv2.GaussianBlur(image_copy, (11,11), 0, cv2.BORDER_CONSTANT)
display("Gauss Noise reduced", image_copy, 604, 806)
cv2.waitKey(0)

image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
image_contour = image_copy.copy()
image_persp = image_copy.copy()
image_copy_A = image_copy.copy()

def convolve2D(image, kernel, padding=0, strides=1, correlation=0, normalize = 0):
    #Correlation
    if correlation == 0:
      kernel = np.flipud(np.fliplr(kernel))
    
    image = np.array(image)

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        #print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                        
                except:
                    break
    if normalize:
      cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
    return output

# Edge Detection
laplacian_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
canny_result = cv2.Canny(image_gray, 50, 200)
ED_type = input("Enter edge detector type Laplace or Canny (L/C)").lower()
if ED_type == 'c':
    display("Canny Edge", canny_result, 604, 806)
    cv2.waitKey(0)
else :
    laplacian_result = convolve2D(image_gray, laplacian_kernel, 1)
    display("Laplacian Edge", laplacian_result, 604, 806)
    cv2.waitKey(0)

#contours
contours, hierarchy = cv2.findContours(canny_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_copy_A, contours, -1, (0,255,0), 3)
display("All Contour", image_copy_A, 604, 806)
cv2.waitKey(0)

hull_list_map = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list_map.append(hull)

cnts = sorted(hull_list_map, key = cv2.contourArea, reverse = True)[:7]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    screenCnt = approx
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image_contour, [screenCnt], -1, (0,255,0), 3)
display("Page Contour", image_contour, 604, 806)
cv2.waitKey(0)

# perspective transformation
width = image_copy.shape[0]
height = image_copy.shape[1]
cnt_pts = np.float32(screenCnt)
transform_pts = np.float32([[width, height],[0, height], [0, 0], [width, 0]])
trans_mat = cv2.getPerspectiveTransform(cnt_pts,transform_pts)
image_persp = cv2.warpPerspective(image_, trans_mat, (width,height))
display("Correct Persp", image_persp, 604, 806)
cv2.waitKey(0)

# adding brightness and contrast
alpha = 2
beta = -150
image_bnc = cv2.addWeighted(image_persp, alpha, np.zeros(image_persp.shape, image_persp.dtype), 0, beta)
display("Brightness and Contrast", image_bnc, 604, 806)
cv2.waitKey(0)

# rotate and size
ang = float(input("enter rotation angle or press enter") or "0")
size = float(input("enter size or press enter") or "1")
cx, cy = int(image_bnc.shape[0]/2), int(image_bnc.shape[1]/2)
Rot_mat = cv2.getRotationMatrix2D(center=(cx,cy), angle=ang, scale=size)
RnS_image = cv2.warpAffine(src=image_bnc, M=Rot_mat, dsize=(image_bnc.shape[1], image_bnc.shape[0]))
display("Rotated", RnS_image, 604, 806)
cv2.waitKey(0)
cv2.imwrite("Doc_scan.jpg", RnS_image)          #Saving the edited document