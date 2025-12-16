import numpy as np
import cv2
import sys

input_file = sys.argv[1]
image = cv2.imread(input_file)
cv2.imshow("img", image)
# gbr_confidence = np.array([[0], [0], [0]])

total_gmb = 0
total_2gmrb = 0

for i in range(image.shape[0]) :
    for j in range(image.shape[1]):
        blu, grn, red = (image[i][j]).astype(int)
        gmb = abs(grn - blu)
        mrb = abs(2*grn - blu - red)
        total_gmb += gmb
        total_2gmrb += mrb
        # for k in range(3):
        #     if gmb >= range_gmb[k][0] and gmb <= range_gmb[k][1]:
        #         if mrb >= range_2gmrmb[k][0] and mrb <= range_2gmrmb[k][1]:
        #             gbr_confidence[k] += 1

total_pix = image.shape[0]*image.shape[1]
avg_gmb = total_gmb/total_pix
avg_2gmrb = total_2gmrb/total_pix
ans = 0
name = ''
if avg_gmb >= 0 and avg_gmb <= 80:
    if avg_2gmrb >= 12 and avg_2gmrb <= 85:
        ans = 2 # grass
        name = 'Grass'

if avg_gmb >= 0 and avg_gmb <= 6:
    if avg_2gmrb >= 1 and avg_2gmrb <= 8:
        ans = 1 # building
        name = 'Building'

if avg_gmb >= 7 and avg_gmb <= 20:
    if avg_2gmrb >= 0 and avg_2gmrb <= 12:
        ans = 3 # road
        name = 'Road'

# print(np.argmax(gbr_confidence))
print("Building = 1, Grass = 2, Road = 3")
print(ans, name)
cv2.waitKey(0)