import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import time


def mask_detection(camera=None,img_path=None):
    model = load_model('masknet.h5')
    media = mp.solutions.face_detection
    detector = media.FaceDetection(min_detection_confidence=0.2)
    mask_label = {0: 'MASK', 1: 'NO MASK'}
    color_label = {0: (0, 255, 0), 1: (0, 0, 255)}
    if camera==True:
        Ptime = 0
        camera = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, frames = camera.read()
            # convert the frame mood to rgb
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            # detect faces
            faces = detector.process(gray)
            # compute FPS
            Ctime = time.time()
            fps = 1 / (Ctime - Ptime)
            Ptime = Ctime
            cv2.putText(frames, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

            if faces.detections: #if there are faces
                for id, detection in enumerate(faces.detections):
                    BBox = detection.location_data.relative_bounding_box
                    h, w, c = gray.shape
                    (x, y, w, h) = int(BBox.xmin * w), int(BBox.ymin * h), int(BBox.width * w), int(BBox.height * h)# get the x y w h
                    # crop the faces from the image
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
    else:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.process(img)  # returns a list of (x,y,w,h) tuples

        if faces.detections:
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

if __name__=="__main__":
    mask_detection(camera=True)