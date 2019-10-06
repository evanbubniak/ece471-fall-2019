import sys
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

import wide_residual_network as wrn
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "cifar10"

if dataset == "cifar100":
    print("cifar100: \n")
    from tensorflow.keras.datasets import cifar100 as cifar
else:
    print("cifar10: \n")
    from tensorflow.keras.datasets import cifar10 as cifar

RANDOM_SEED = 31415
BATCH_SIZE = 100
NUM_EPOCHS = 100

(training_X, training_Y), (testX, testY) = cifar.load_data()

training_X = training_X.astype('float32')
training_X = (training_X - training_X.mean(axis=0)) / (training_X.std(axis=0))

testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

training_Y = keras.utils.to_categorical(training_Y)
testY = keras.utils.to_categorical(testY)

trainX, valX, trainY, valY = train_test_split(training_X, training_Y, test_size=1/6, random_state=RANDOM_SEED)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,)

img_shape = (3, 32, 32) if K.image_data_format() == 'channels_first' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
model = wrn.create_wide_residual_network(img_shape, nb_classes=10, N=2, k=8, dropout=0.00)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc", "top_k_categorical_accuracy"])
print("Finished compiling")

#model.load_weights("weights/WRN-16-8 Weights.h5")
print("Model loaded.")

model.fit_generator(generator.flow(trainX, trainY, BATCH_SIZE=BATCH_SIZE), steps_per_epoch=len(trainX) // BATCH_SIZE, epochs=NUM_EPOCHS,
                   callbacks=[callbacks.ModelCheckpoint("WRN-16-8 Weights.h5",
                                                        monitor="val_acc",
                                                        save_best_only=True,
                                                        verbose=1)],
                   validation_data=(valX, valY),
                   validation_steps=valX.shape[0] // BATCH_SIZE,)

model.evaluate(testX, testY, verbose = 1)
#yPreds = model.predict(testX)
# yPred = np.argmax(yPreds, axis=1)
# yPred = keras.utils.to_categorical(yPred)
# yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

