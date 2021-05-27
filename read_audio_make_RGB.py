import numpy as np
import pandas as pd
import librosa
import os
import pywt
import cv2 as cvlib
from args import parser
import matplotlib.pyplot as plt

args = parser.parse_args()

def normalize_data_all_gather(vect_in, out_min, out_max, percent_acceptation=80,not_clip_until_acceptation_time_factor=1.5):
    percent_val = np.percentile(abs(vect_in).reshape((vect_in.shape[0], vect_in.shape[1] * vect_in.shape[2])),percent_acceptation, axis=1)
    percent_val_matrix = not_clip_until_acceptation_time_factor * np.repeat(percent_val,vect_in.shape[1] * vect_in.shape[2],axis=0).reshape((vect_in.shape[0], vect_in.shape[1], vect_in.shape[2]))
    matrix_clip = np.maximum(np.minimum(vect_in, percent_val_matrix), -percent_val_matrix)
    return np.divide(matrix_clip, percent_val_matrix) * ((out_max - out_min) / 2) + (out_max + out_min) / 2

x=10971
#path of files where images should be saved
path_liste_file= args.audio_listefile_folder_path
data = pd.read_csv(path_liste_file)

#making the numpy array for RGB
RGB_sound1 = np.zeros((3200, 299, 299, 3))

n_fft = 40
hop_length = 20

path= args.audio_folder_path
path_save= args.audio_images_folder_path
print("-------Loop for RGB1---------")
for i in range(3200) :
  print('loop #',i)
  print('file name',data['filename'].iloc[i])
  print('target no', data['target'].iloc[i])
  print('target category', data['category'].iloc[i])
  namefile = data['filename'].iloc[i]
  print(namefile)
  filepath = os.path.join(path, namefile)
  clipnoise, sample_rate = librosa.load(filepath, duration=8.0)
  scales = np.arange(1, 128)
  waveletname = 'morl'
  coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
  scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
  stft_magnitude, stft_phase = librosa.magphase(stft)
  stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
  spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)
  mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  RGB_sound1[i, :, :, 0] = spectrogramimg
  RGB_sound1[i, :, :, 1] = scalogramimg
  RGB_sound1[i, :, :, 2] = mfccimg
  print('finished loop:', i)
print("------------------------------------------------------")
#normalising the RGB data
RGB_sound1[:, :, :, 0] = normalize_data_all_gather(RGB_sound1[:, :, :, 0], -1, 1, 95, 2)
RGB_sound1[:, :, :, 1] = normalize_data_all_gather(RGB_sound1[:, :, :, 1], -1, 1, 95, 2)
RGB_sound1[:, :, :, 2] = normalize_data_all_gather(RGB_sound1[:, :, :, 2], -1, 1, 95, 2)

#saving the RGB data
path_save_rgb= os.path.join(path_save, 'RGB_sound1')
np.save(path_save_rgb,RGB_sound1)
del RGB_sound1
#making the numpy array for RGB
RGB_sound2 = np.zeros((3200, 299, 299, 3))

path= args.audio_folder_path
path_save= args.audio_images_folder_path
print("-------Loop for RGB2---------")
for i in range(3200,6400) :
  print('loop #',i)
  print('file name',data['filename'].iloc[i])
  print('target no', data['target'].iloc[i])
  print('target category', data['category'].iloc[i])
  namefile = data['filename'].iloc[i]
  print(namefile)
  filepath = os.path.join(path, namefile)
  clipnoise, sample_rate = librosa.load(filepath, duration=8.0)
  scales = np.arange(1, 128)
  waveletname = 'morl'
  coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
  scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
  stft_magnitude, stft_phase = librosa.magphase(stft)
  stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
  spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)
  mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  RGB_sound2[i-3200, :, :, 0] = spectrogramimg
  RGB_sound2[i-3200, :, :, 1] = scalogramimg
  RGB_sound2[i-3200, :, :, 2] = mfccimg
  print('finished loop:', i)
print("------------------------------------------------------")
#normalising the RGB data
RGB_sound2[:, :, :, 0] = normalize_data_all_gather(RGB_sound2[:, :, :, 0], -1, 1, 95, 2)
RGB_sound2[:, :, :, 1] = normalize_data_all_gather(RGB_sound2[:, :, :, 1], -1, 1, 95, 2)
RGB_sound2[:, :, :, 2] = normalize_data_all_gather(RGB_sound2[:, :, :, 2], -1, 1, 95, 2)

#saving the RGB data
path_save_rgb= os.path.join(path_save, 'RGB_sound2')
np.save(path_save_rgb,RGB_sound2)
del RGB_sound2

#making the numpy array for RGB
RGB_sound3 = np.zeros((3200, 299, 299, 3))
n_fft = 40
hop_length = 20
path= args.audio_folder_path
path_save= args.audio_images_folder_path
print("-------Loop for RGB3---------")
for i in range(6400,9600) :
  print('loop #',i)
  print('file name',data['filename'].iloc[i])
  print('target no', data['target'].iloc[i])
  print('target category', data['category'].iloc[i])
  namefile = data['filename'].iloc[i]
  print(namefile)
  filepath = os.path.join(path, namefile)
  clipnoise, sample_rate = librosa.load(filepath, duration=8.0)
  scales = np.arange(1, 128)
  waveletname = 'morl'
  coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
  scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
  stft_magnitude, stft_phase = librosa.magphase(stft)
  stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
  spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)
  mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  RGB_sound3[i-6400, :, :, 0] = spectrogramimg
  RGB_sound3[i-6400, :, :, 1] = scalogramimg
  RGB_sound3[i-6400, :, :, 2] = mfccimg
  print('finished loop:', i)
print("------------------------------------------------------")
#normalising the RGB data
RGB_sound3[:, :, :, 0] = normalize_data_all_gather(RGB_sound3[:, :, :, 0], -1, 1, 95, 2)
RGB_sound3[:, :, :, 1] = normalize_data_all_gather(RGB_sound3[:, :, :, 1], -1, 1, 95, 2)
RGB_sound3[:, :, :, 2] = normalize_data_all_gather(RGB_sound3[:, :, :, 2], -1, 1, 95, 2)

#saving the RGB data
path_save_rgb= os.path.join(path_save, 'RGB_sound3')
np.save(path_save_rgb,RGB_sound3)
del RGB_sound3
#making the numpy array for RGB
RGB_sound4 = np.zeros((1371, 299, 299, 3))
n_fft = 40
hop_length = 20
path= args.audio_folder_path
path_save= args.audio_images_folder_path
print("-------Loop for RGB4---------")
for i in range(9600,10971) :
  print('loop #',i)
  print('file name',data['filename'].iloc[i])
  print('target no', data['target'].iloc[i])
  print('target category', data['category'].iloc[i])
  namefile = data['filename'].iloc[i]
  print(namefile)
  filepath = os.path.join(path, namefile)
  clipnoise, sample_rate = librosa.load(filepath, duration=8.0)
  scales = np.arange(1, 128)
  waveletname = 'morl'
  coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
  scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
  stft_magnitude, stft_phase = librosa.magphase(stft)
  stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
  spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)
  mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
  RGB_sound4[i-9600, :, :, 0] = spectrogramimg
  RGB_sound4[i-9600, :, :, 1] = scalogramimg
  RGB_sound4[i-9600, :, :, 2] = mfccimg
  print('finished loop:', i)
print("------------------------------------------------------")
#normalising the RGB data
RGB_sound4[:, :, :, 0] = normalize_data_all_gather(RGB_sound4[:, :, :, 0], -1, 1, 95, 2)
RGB_sound4[:, :, :, 1] = normalize_data_all_gather(RGB_sound4[:, :, :, 1], -1, 1, 95, 2)
RGB_sound4[:, :, :, 2] = normalize_data_all_gather(RGB_sound4[:, :, :, 2], -1, 1, 95, 2)

#saving the RGB data
path_save_rgb= os.path.join(path_save, 'RGB_sound4')
np.save(path_save_rgb,RGB_sound4)
del RGB_sound4


#making numpy array for y_label
y_label_sound=np.zeros((10971,1))
print("Making y_label data")
for i in range(10971):
  print('loop #',i)
  print('file name',data['filename'].iloc[i])
  print('target no', data['target'].iloc[i])
  print('target category', data['category'].iloc[i])
  y_label_sound[i,:]=data['target'].iloc[i]
print("------------------------------------------")
#saving y_label
path_save_ylabel= os.path.join(path_save, 'y_label_sound')
np.save(path_save_ylabel,y_label_sound)

#numpy array for y_label_hot_sound
y_label_hot_sound=np.zeros((10971,10))
print("Making y_label_hot_sound data")
for i in range(10971) :
  print('loop #',i)
  print('file name',data['filename'].iloc[i])
  print('target no', data['target'].iloc[i])
  print('target category', data['category'].iloc[i])
  y_label_hot_sound[i, data['target'].iloc[i]] = 1
print("------------------------------------------")
#saving y_label_hot_sound
path_save_ylabel_hot= os.path.join(path_save, 'y_label_hot_sound')
np.save(path_save_ylabel_hot,y_label_hot_sound)

rgb1=np.load("/content/drive/MyDrive/Prism_worklet/audio_images/RGB_sound1.npy",mmap_mode='r+')
rgb2=np.load("/content/drive/MyDrive/Prism_worklet/audio_images/RGB_sound2.npy",mmap_mode='r+')
rgb3=np.load("/content/drive/MyDrive/Prism_worklet/audio_images/RGB_sound3.npy",mmap_mode='r+')
rgb4=np.load("/content/drive/MyDrive/Prism_worklet/audio_images/RGB_sound4.npy",mmap_mode='r+')
merge_arr = np.concatenate([rgb1,rgb2,rgb3,rgb4], axis=0)
print(merge_arr.shape)

path_save_rgb="/content/drive/MyDrive/Prism_worklet/audio_images/RGB_sound"
np.save(path_save_rgb,merge_arr)

print('saved to disk')
