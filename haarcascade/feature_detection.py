import cv2
import numpy as np

img = cv2.imread("dexmo.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("dexmo_flip.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(img1, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

matching_result = cv2.drawMatches(img, kp1, img1, kp2, matches[:30], None, flags=2)

cv2.imshow("img", img)
cv2.imshow("img1", img1)
cv2.imshow("matching_result", matching_result)


cv2.waitKey(0)
cv2.destroyAllWindows()