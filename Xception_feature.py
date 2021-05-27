
import numpy as np
import os
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.applications import Xception
from args import parser
print("Reading paths......")
args = parser.parse_args()
path_save_Xception= args.Xception_features_path
print("Loading Files......")
path_audio_img= args.audio_images_folder_path
path_save_rgb= os.path.join(path_audio_img, 'RGB_sound.npy')
RGB_sound=np.load(path_save_rgb,mmap_mode='r+')

print('shape of RGB sound',RGB_sound.shape)
print("Extracting Features.......")
model = Xception(weights='imagenet', include_top=False)
test1=RGB_sound[:4000,:,:,:]
preds = model.predict(test1)
path_save_feat="/content/drive/MyDrive/Prism_worklet/Xception_features/feature1"
print('pred_reduc taille',preds.shape)
np.save(path_save_feat,preds)
del preds
del test1
test2=RGB_sound[4000:8000,:,:,:]
preds = model.predict(test2)
path_save_feat="/content/drive/MyDrive/Prism_worklet/Xception_features/feature2"
print('pred_reduc taille',preds.shape)
np.save(path_save_feat,preds)
del preds
del test2
test3=RGB_sound[8000:10971,:,:,:]
preds = model.predict(test3)
path_save_feat="/content/drive/MyDrive/Prism_worklet/Xception_features/feature3"
print('pred_reduc taille',preds.shape)
np.save(path_save_feat,preds)
del preds
del test3
feat1=np.load("/content/drive/MyDrive/Prism_worklet/Xception_features/feature1.npy",mmap_mode='r+')
feat2=np.load("/content/drive/MyDrive/Prism_worklet/Xception_features/feature2.npy",mmap_mode='r+')
feat3=np.load("/content/drive/MyDrive/Prism_worklet/Xception_features/feature3.npy",mmap_mode='r+')
merge_arr = np.concatenate([feat1,feat2,feat3], axis=0)
print(merge_arr.shape)
pred_reduc_pool = merge_arr.reshape((-1,2048 * 10 * 10))

print('concatenate in size',pred_reduc_pool.shape)
print("Saving Features........")
path_save_features= os.path.join(path_save_Xception, 'Xception_feat')

np.save(path_save_features,pred_reduc_pool)
print('ended successfully')
