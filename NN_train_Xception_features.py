import numpy as np
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
# from args import parser
import os
import time
from sklearn.model_selection import train_test_split

# args = parser.parse_args()

print("Keras version:",keras.__version__)
print("Tensorflow version:",tensorflow.__version__)

feat_input=np.load("/content/drive/MyDrive/Prism_worklet/Xception_features/Xception_feat.npy")
y_label_sound=np.load("/content/drive/MyDrive/Prism_worklet/audio_images/y_label_sound.npy")
y_label_hot_sound=np.load("/content/drive/MyDrive/Prism_worklet/audio_images/y_label_hot_sound.npy")

print('feat input shape',feat_input.shape)
print('label shape',y_label_hot_sound.shape)

modeltop = Sequential()

# in the first layer, you must specify the expected input data shape:
modeltop.add(Dense(1024, activation='sigmoid', input_dim=204800))
modeltop.add(Dropout(0.3))
modeltop.add(Dense(512, activation='relu'))
modeltop.add(Dropout(0.2))
modeltop.add(Dense(256, activation='relu'))
modeltop.add(Dropout(0.15))
modeltop.add(Dense(64, activation='relu'))
modeltop.add(Dropout(0.1))
modeltop.add(Dense(32, activation='relu'))
modeltop.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.001, decay=1e-5)

modeltop.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train=feat_input
y_train=y_label_hot_sound

x_train_cv,x_valid_cv,y_train_cv,y_valid_cv=train_test_split(x_train,y_train, test_size=0.3, random_state=200)

print(x_train_cv.shape,x_valid_cv.shape,y_train_cv.shape,y_valid_cv.shape)

modeltop.fit(x_train_cv, y_train_cv, epochs=150, batch_size=128, shuffle=True, verbose=1, validation_data=(x_valid_cv, y_valid_cv),callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=20),keras.callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_loss',save_best_only=True)])
print("Evaluating on test set:-",modeltop.evaluate(x_valid_cv, y_valid_cv))


#saving the trained model to disk
model_json = modeltop.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
modeltop.save_weights("model2.h5")
print("Saved model to disk")