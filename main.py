import cv2
import numpy as np
import face_recognition as fr
import os 
from datetime import datetime 

# source_image
path = 'students'

std_imgs = []
std_img_namelist = os.listdir(path)
std_namelist = []

for img in std_img_namelist:
    cur_img = cv2.imread(f'{path}/{img}')
    std_imgs.append(cur_img)
    std_namelist.append(os.path.splitext(img)[0])

def encode_imgs(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

encoded_std_imgs = encode_imgs(std_imgs)
# print(encoded_std_imgs[0])
print(f'Encoding completed for {len(encoded_std_imgs)} students.')


camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    img_frame = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

    faces_in_frame = fr.face_locations(img_frame)
    en_faces_in_frame = fr.face_encodings(img_frame, faces_in_frame)

    for  encodedface, facelocation in zip(en_faces_in_frame, faces_in_frame):
        matches = fr.compare_faces(encoded_std_imgs, encodedface)
        facedist = fr.face_distance(encoded_std_imgs, encodedface)
        matchIndex = np.argmin(facedist)
    
        if matches[matchIndex]:
            std_name = std_namelist[matchIndex].upper()

            x1, x2, y1, y2 = facelocation
            x1, x2, y1, x2 = x1*4, x2*4, y1*4, y2*4

            cv2.rectangle(frame, (y2,x1),(x2,y1),(0,255,0),2)
            cv2.rectangle(frame, (y2,x1-35), (x2,y1),(0, 255, 0), cv2.FILLED)
            cv2.putText(frame, std_name, (y2+6, y1-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            print(std_name)
            
    cv2.imshow('camera frames', frame)
    cv2.waitKey(0) 

cv2.destroyAllWindows()
camera.release()

