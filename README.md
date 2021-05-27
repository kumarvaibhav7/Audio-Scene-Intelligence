# SRMIST_VI32SRM_Audio_Scene_Intelligence_Domestic
SRIB-PRISM Program

Link of Dataset used: https://drive.google.com/drive/folders/1AD3FQOcTeNWHbalkOUOOH7DtcfFJC5jm?usp=sharing
Link of meta file for dataset: https://docs.google.com/spreadsheets/d/1FKjPVUe8SiM5-DZNpvFQsWqzW-YEiu-zZTMQnFDEUAE/edit?usp=sharing

Files with their respective functions:
1. args.py- contains path of where temporary files should be saved. So that it can be used in another part.
2. read_audio_make_RGB.py- Preprocesses the audio files and converts it to a array of RGB images (Spectrogram, Scalogram and MFCC) and also stores the labels of audio in a numpy array.
3. Xception_features.py- Extract the features from the RGB images using prebuilt Xception Model with imagenet weights and convert it to a vector and save the feature vector to a numpy array.
4. NN_train_Xception_features.py- Train a deep neural network on the feature vector provided from the previous section and save the trained model on disk.
5. model_predict.py- Predict the class of a audio with trained neural network by giving the path in input.
6. prism_final.ipynb- Notebook to execute the code.