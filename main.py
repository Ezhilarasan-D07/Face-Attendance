import cv2
import face_recognition as fr
import os
from datetime import datetime
import numpy as np

# loading an image 
img_sam = fr.load_image_file('students/VJS.jpg')
img_sam = cv2.resize(img_sam, (0,0), None, 0.25, 0.25)
img_sam = cv2.cvtColor(img_sam, cv2.COLOR_BGR2RGB)

# original image 
locate_sam=fr.face_locations(img_sam)[0]
encode_sam = fr.face_encodings(img_sam)[0]
cv2.rectangle(img_sam,(locate_sam[0], locate_sam[2]*2),(locate_sam[1], locate_sam[3]*2),(255,0,0),2)

# print(len(encode_sam))
# testing or comparing images  
img_test = fr.load_image_file('students/Samantha.jpg')
img_test = cv2.resize(img_test, (0,0), None, 0.25, 0.25)
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# 
locate_test = fr.face_locations(img_test)[0]
encode_test = fr.face_encodings(img_test)[0]
cv2.rectangle(img_test,(locate_test[0], locate_test[2]*2),(locate_test[1], locate_test[3]*2),(255,0,0),2)

compare_faces = fr.compare_faces([encode_sam],encode_test)
compare_dis = fr.face_distance([encode_sam],encode_test)
print(compare_faces, compare_dis )

cv2.imshow('frame', img_sam)
cv2.imshow('test', img_test)
cv2.waitKey(0)



