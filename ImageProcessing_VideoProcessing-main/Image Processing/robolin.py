import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# input_file = sys.argv[1]
# image = cv2.imread(input_file)
image = cv2.imread("Image Processing/EE604 A2/tiles8.jpg")

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(src=img_gray, ksize=(9, 9), sigmaX=3)
canny_img = cv2.Canny(img_gray, 10, 25)
# creating function for hough transform
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

# creating function for detecting the peaks
def hough_peaks(H, num_peaks, threshold=0, nhood_size=3):
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)
        if(idx<threshold):
            continue
        H1_idx = np.unravel_index(idx, H1.shape)
        indicies.append(H1_idx)
        idx_y, idx_x = H1_idx

        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):

                H1[y, x] = 0
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    return indicies, H

# function for plotting the hough transform
def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)
    	
    plt.imshow(H, cmap='jet')
    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()

#function for drawing lines on image
def hough_lines_draw(img, indicies, rhos, thetas):
    for i in range(len(indicies)):

        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


H, rhos, thetas = hough_lines_acc(canny_img)
indicies, H = hough_peaks(H, 10, nhood_size=21) # finding peaks
plot_hough_acc(H) # plotting the hough transform
hough_lines_draw(image, indicies, rhos, thetas) #drawing the lines on original image

cv2.imshow("Marked", image)
status = cv2.imwrite('robolin-tiles3.jpg', image)
print("Image written to file-system : ", status)
cv2.waitKey(0)
cv2.destroyAllWindows()