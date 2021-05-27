import numpy as np
import pandas as pd
import librosa
import os
import pywt
import cv2 as cvlib
from args import parser
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.applications import Xception
from keras.models import model_from_json

def normalize_data_all_gather(vect_in, out_min, out_max, percent_acceptation=80,not_clip_until_acceptation_time_factor=1.5):
    percent_val = np.percentile(abs(vect_in).reshape((vect_in.shape[0], vect_in.shape[1] * vect_in.shape[2])),percent_acceptation, axis=1)
    percent_val_matrix = not_clip_until_acceptation_time_factor * np.repeat(percent_val,vect_in.shape[1] * vect_in.shape[2],axis=0).reshape((vect_in.shape[0], vect_in.shape[1], vect_in.shape[2]))
    matrix_clip = np.maximum(np.minimum(vect_in, percent_val_matrix), -percent_val_matrix)
    return np.divide(matrix_clip, percent_val_matrix) * ((out_max - out_min) / 2) + (out_max + out_min) / 2

filepath=input("Enter File path:-")
print("---------------------------------------")
print("opening file")
n_fft = 40
hop_length = 20
clipnoise, sample_rate = librosa.load(filepath, duration=8.0)
scales = np.arange(1, 128)
waveletname = 'morl'
print("preprocessing file")
coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
stft_magnitude, stft_phase = librosa.magphase(stft)
stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)
mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
print("making RGB image")
RGB_test=np.zeros((1, 299, 299, 3))
RGB_test[0, :, :, 0] = spectrogramimg
RGB_test[0, :, :, 1] = scalogramimg
RGB_test[0, :, :, 2] = mfccimg

RGB_test[:, :, :, 0] = normalize_data_all_gather(RGB_test[:, :, :, 0], -1, 1, 95, 2)
RGB_test[:, :, :, 1] = normalize_data_all_gather(RGB_test[:, :, :, 1], -1, 1, 95, 2)
RGB_test[:, :, :, 2] = normalize_data_all_gather(RGB_test[:, :, :, 2], -1, 1, 95, 2)

print("Extracting Xception features")
model = Xception(weights='imagenet', include_top=False)
preds = model.predict(RGB_test)

pred_reduc_pool = preds.reshape((-1,2048 * 10 * 10))
print("Predicting Results")
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("best_model.h5")
print("Loaded model from disk")

predictions = loaded_model.predict(pred_reduc_pool)
print(predictions)
classes={1:'absence',2:'cooking',3:'dishwashing',4:'eating',5:'other',6:'social_activity',7:'vacuum_cleaner',8:'watching_tv',9:'working'}
index=predictions[0].argmax(axis=0)
print("The predicted class is:-",classes[index])
print("With confidence:-",(predictions[0][index]*100),"%")