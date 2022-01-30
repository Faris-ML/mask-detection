from keras.applications.vgg19 import VGG19
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


# the data path -change the path for your own data-
train_dir = 'Face Mask Dataset\Train'
val_dir = 'Face Mask Dataset\Validation'

# make a augmentation generator for training data
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2,brightness_range=[0.5,1.5])
train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(128, 128,),class_mode='categorical',batch_size=32)

# make a augmentation generator for validation data
val_datagen = ImageDataGenerator(rescale=1.0/255,horizontal_flip=True, zoom_range=0.2,shear_range=0.2,brightness_range=[0.5,1.5])
val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128, 128,),class_mode='categorical',batch_size=32)

# load VVG19 architecture
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in vgg19.layers:
    layer.trainable = False


# build the model architecture and add some layers
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add (Dense(64,activation='relu'))
model.add (Dense(32,activation='relu'))
model.add (Dense(16,activation='relu'))
model.add (Dense(8,activation='relu'))
model.add (Dense(4,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")
history = model.fit_generator(generator=train_generator,
                              epochs=20,validation_data=val_generator
                              )

model.save('masknet.h5')