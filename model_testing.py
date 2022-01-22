from keras.models import load_model
from keras.preprocessing.image import image_dataset_from_directory
import numpy as np
from sklearn.metrics import confusion_matrix
model=load_model('masknet.h5')
test_dir = 'Face Mask Dataset\Test'

test= image_dataset_from_directory(directory='Face Mask Dataset\Test',label_mode='categorical',batch_size=992,image_size=(128,128))

y_pred=np.array([])
y_true=np.array([])

for x,y in test.take(1):
    y_pred=np.concatenate([y_pred,np.argmax(model.predict(test),axis=-1)])
    y_true=np.concatenate([y_true,np.argmax(y.numpy(),axis=-1)])
cm=confusion_matrix(y_true=y_true,y_pred=y_pred)
print(cm)
