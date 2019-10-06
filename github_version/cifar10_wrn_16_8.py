import numpy as np
import sklearn.metrics as metrics

import wide_residual_network as wrn
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend as K

batch_size = 100
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = keras.utils.to_categorical(trainY)
testY = keras.utils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,)

init_shape = (3, 32, 32) if K.image_data_format() == 'channels_first' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.00)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
print("Finished compiling")

#model.load_weights("weights/WRN-16-8 Weights.h5")
print("Model loaded.")

model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                   callbacks=[callbacks.ModelCheckpoint("weights/WRN-16-8 Weights.h5",
                                                        monitor="val_acc",
                                                        save_best_only=True,
                                                        verbose=1)],
                   validation_data=(testX, testY),
                   validation_steps=testX.shape[0] // batch_size,)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yPred = keras.utils.to_categorical(yPred)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

