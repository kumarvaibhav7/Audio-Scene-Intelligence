import argparse

parser = argparse.ArgumentParser(description='Audio classification')

parser.add_argument('--audio_listefile_folder_path', default='/content/drive/MyDrive/DCASE2018-task5-dev/meta1.csv', type=str)
parser.add_argument('--audio_folder_path', default='/content/drive/MyDrive/DCASE2018-task5-dev', type=str)
parser.add_argument('--audio_images_folder_path', default='/content/drive/MyDrive/Prism_worklet/audio_images/', type=str)

parser.add_argument('--Xception_features_path', default='/content/drive/MyDrive/Prism_worklet/Xception_features/', type=str)



