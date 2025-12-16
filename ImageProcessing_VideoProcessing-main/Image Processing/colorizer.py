import cv2
import numpy as np
import sys

input1 = sys.argv[1]
input2 = sys.argv[2]
input3 = sys.argv[3]
cb = cv2.imread(input2, 0)
cr = cv2.imread(input3, 0)
imY = cv2.imread(input1, 0)

upupcr = cv2.resize(cr, [imY.shape[1], imY.shape[0]], interpolation = cv2.INTER_AREA)
upupcb = cv2.resize(cb, [imY.shape[1], imY.shape[0]], interpolation = cv2.INTER_AREA)
d, s, c = 77, 40, 60
upupcr = cv2.bilateralFilter(upupcr, d, s, c)
upupcb = cv2.bilateralFilter(upupcb, d, s, c)
imY = cv2.bilateralFilter(imY, d, s, s)

# R = imY + 1.402 * (upupcr - 128) 
# G = imY + 1.772 * (upupcb - 128)
# B = imY - 0.34414 * (upupcb - 128) - 0.71414 * (upupcr- 128)

B = np.zeros(imY.shape)
G = np.zeros(imY.shape)
R = np.zeros(imY.shape)
for i in range(imY.shape[0]) :
    for j in range(imY.shape[1]):
        R[i][j] = imY[i][j] + 1.402 * (upupcr[i][j] - 128) 
        G[i][j] = imY[i][j] + 1.772 * (upupcb[i][j] - 128)
        B[i][j] = imY[i][j] - 0.34414 * (upupcb[i][j] - 128) - 0.71414 * (upupcr[i][j]- 128)

res = []
res.append(np.uint8(G))
res.append(np.uint8(B))
res.append(np.uint8(R))
res = cv2.merge(res)

cv2.imshow("res", res)
status = cv2.imwrite('flyingelephant.jpg', res)
print("Image written to file-system : ", status)

cv2.waitKey(0)
cv2.destroyAllWindows()