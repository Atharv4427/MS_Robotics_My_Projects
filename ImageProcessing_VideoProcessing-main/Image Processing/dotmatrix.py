import cv2
import numpy as np

blank_image = np.zeros((300,500,3), np.uint8) # creating blank image of size 300 x 500
digit = np.mgrid[55:186:65, 30:271:60].reshape(2,-1).T  # defining the location of centers of the disks

# function to print the discs to image
def make_digit(img, offset, num, digit):
    swon = np.array([0, 10, 4, 14])
    if(num == 0):
        swon = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14])
    if(num == 1):
        swon = np.array([10, 11, 12, 13, 14])
    if(num == 2):
        swon = np.array([0, 5, 10, 11, 12, 7, 2, 3, 4, 9, 14])
    if(num == 3):
        swon = np.array([0, 5, 10, 11, 12, 7, 2, 13, 4, 9, 14])
    if(num == 4):
        swon = np.array([0, 10, 11, 12, 7, 2, 13, 14, 1])
    if(num == 5):
        swon = np.array([0, 5, 10, 1, 2, 7, 12, 13, 4, 9, 14])
    if(num == 6):
        swon = np.array([0, 5, 10, 1, 2, 7, 12, 13, 4, 9, 14, 3])
    if(num == 7):
        swon = np.array([0, 5, 10, 12, 13, 14, 11])
    if(num == 8):
        swon = np.array([0, 5, 10, 1, 2, 7, 12, 13, 4, 9, 14, 3, 11])
    if(num == 9):
        swon = np.array([0, 5, 10, 1, 2, 7, 12, 13, 4, 9, 14, 11])
    for i in swon:
        cv2.circle(img, (digit[i][0] + offset, digit[i][1]), 25, (255, 255, 255), -1)
    return img

# function to print the input number to the dot matrix image
def make_num(img, num, digit):
    a = int(num/10)
    make_digit(img, 0, a, digit)
    a = num%10
    # x_offset = 260
    make_digit(img, 260, a, digit)
    return img

n = int(input("Enter Number Between 0 and 99: "))
digit_image = make_num(blank_image, n, digit)
cv2.imshow("dot_matrix", digit_image)
status = cv2.imwrite('dotmatrix.jpg', digit_image)
print("Image written to file-system : ", status)
cv2.waitKey(0)
cv2.destroyAllWindows()