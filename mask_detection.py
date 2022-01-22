import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import time

def face_distance(min_distance,face1,face2):
    label=0
    P1=[(face1[0]-face1[2]),(face1[2]-face1[3])]
    P2=[(face1[0]+face1[2]),(face1[2]-face1[3])]
    P3=[(face1[0]-face1[2]),(face1[2]+face1[3])]
    vector1=[P2[0]-P1[0],P2[1]-P1[1]]
    vector2=[P3[0]-P1[0],P3[1]-P1[1]]
    face1_area=np.linalg.norm(np.cross(np.array(vector1),np.array(vector2)))
    P1 = [(face2[0] - face2[2]), (face2[2] - face2[3])]
    P2 = [(face2[0] + face2[2]), (face2[2] - face2[3])]
    P3 = [(face2[0] - face2[2]), (face2[2] + face2[3])]
    vector1 = [P2[0] - P1[0], P2[1] - P1[1]]
    vector2 = [P3[0] - P1[0], P3[1] - P1[1]]
    face2_area = np.linalg.norm(np.cross(np.array(vector1), np.array(vector2)))
    ratio=min(face2_area,face1_area)/max(face2_area,face1_area)
    if ratio < 0.2:
        label=0
    else:
        vector=np.array([(face1[0]-face2[0]),(face1[1]-face2[1])])
        distance=np.linalg.norm(vector)
        if distance < min_distance:
            label=1
        else:
            label=0
    return label



media=mp.solutions.face_detection
draw=mp.solutions.drawing_utils
detector=media.FaceDetection(min_detection_confidence=0.2)

#trying it out on a sample image
img = cv2.imread('dataset\\maksssksksss101.png')
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
faces = detector.process(img) #returns a list of (x,y,w,h) tuples
print(faces)


MIN_DISTANCE = 100

#Load train and test set

model=load_model('masknet.h5')


mask_label = {0:'MASK',1:'NO MASK'}
color_label = {0:(0,255,0),1:(0,0,255)}
if faces.detections:
    label = [0 for i in range(len(faces.detections))]
    for id, detection in enumerate(faces.detections):
        BBox = detection.location_data.relative_bounding_box
        h, w, c = img.shape
        (x, y, w, h) = int(BBox.xmin * w), int(BBox.ymin * h), int(BBox.width * w), int(BBox.height * h)
        crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        cv2.putText(img, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color_label[mask_result.argmax()], 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color_label[mask_result.argmax()], 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()
else:
    print("No. of faces detected is less than 1")


Ptime=0
camera=cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame

    ret, frames = camera.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    faces=detector.process(gray)

    #label = [0 for i in range(len(faces))]
    # Draw a rectangle around the faces
    Ctime=time.time()
    fps=1/(Ctime-Ptime)
    Ptime=Ctime
    cv2.putText(frames, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)

    if faces.detections:
        label = [0 for i in range(len(faces.detections))]
        for id,detection in enumerate(faces.detections):
            BBox=detection.location_data.relative_bounding_box
            h,w,c=gray.shape
            (x,y,w,h)=int(BBox.xmin*w),int(BBox.ymin*h),int(BBox.width*w),int(BBox.height*h)
            crop = gray[y:y + h, x:x + w]
            crop = cv2.resize(crop, (128, 128))
            crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
            mask_result = model.predict(crop)
            cv2.putText(frames, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color_label[mask_result.argmax()], 2)
            cv2.rectangle(frames, (x, y), (x + w, y + h), color_label[mask_result.argmax()], 1)
    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()